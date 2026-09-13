"""
shifts.py — the SHIFT SIMULATOR.

The organisers told us the hidden test "deliberately changes lighting,
orientation, backgrounds and defect presentation". Our training images do not
contain those changes, so a model trained on them as-is would meet the test's
conditions for the first time on the day it is marked.

So during training we change every image at random before the model sees it
(this is called DATA AUGMENTATION). Each function below imitates one real-world
change a factory camera would face. The model is shown a differently-disturbed
copy of each part every time round; the only thing that never changes is the
defect itself, so the defect is the only thing it can reliably learn.

Every function takes an image (a 128x128x3 grid of numbers 0-255) and a random
number generator, and returns a new image. Nothing here changes the LABEL: a
scratched part turned upside down is still a scratched part.

The same functions, at fixed strengths, build the STRESS TEST in evaluate.py.
Only numpy and Pillow are used (no OpenCV), to keep the install small.
"""
import io

import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------- helpers

def border_colour(img: np.ndarray) -> np.ndarray:
    """The typical colour of the image's outer frame — i.e. the background colour."""
    frame = np.concatenate([img[:3].reshape(-1, 3), img[-3:].reshape(-1, 3),
                            img[:, :3].reshape(-1, 3), img[:, -3:].reshape(-1, 3)])
    return np.median(frame, axis=0)


def background_mask(img: np.ndarray, tol: float = 18.0):
    """
    Which pixels are background (True) and which are the part (False).

    The background is the area that (a) looks like the frame colour and
    (b) is connected to the edge of the picture. We start from the edge and
    grow inwards one pixel at a time through similar-coloured pixels; the part's
    dark outline stops the growth, so it never leaks inside the part.

    Returns None when the answer does not look trustworthy (for example when
    the background is almost the same grey as the part), so callers skip it.
    """
    diff = np.linalg.norm(img.astype(np.float32) - border_colour(img), axis=2)
    similar = diff < tol
    reached = np.zeros_like(similar)
    reached[0], reached[-1] = similar[0], similar[-1]
    reached[:, 0] |= similar[:, 0]
    reached[:, -1] |= similar[:, -1]
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= similar
        if np.array_equal(grown, reached):
            break
        reached = grown
    share = reached.mean()
    if share < 0.10 or share > 0.75:   # implausible: the part fills roughly half the picture
        return None
    return reached


# ---------------------------------------------------------------- ORIENTATION

def rotate(img, rng, angle=None, scale=None, shift=None):
    """
    Turn the part to ANY angle, move it, and zoom it in or out.
    Training only tilts parts by about ±40°; the hidden test changes orientation.
    The corners uncovered by turning are filled with the background colour, so
    the rotation itself does not add a tell-tale black triangle.

    The zoom range is WIDE (0.6-1.2) because of a bug the stress test found.
    With a narrow range (0.85-1.12) the model called every far-away normal part
    "contamination": once the lighting family stopped it using colour, it
    learned dirt as "small spots" — and from further away, the three normal
    dots ARE small spots. Showing it parts at many distances teaches it that
    size relative to the part is what matters.
    """
    h, w = img.shape[:2]
    angle = rng.uniform(0, 360) if angle is None else angle
    scale = rng.uniform(0.6, 1.2) if scale is None else scale
    tx, ty = rng.uniform(-10, 10, size=2) if shift is None else shift
    t = np.deg2rad(angle)
    cos, sin = np.cos(t), np.sin(t)
    cx, cy = w / 2, h / 2
    # Pillow asks the question backwards: "for each pixel of the OUTPUT, where in
    # the INPUT should I look?" So we hand it the inverse of the movement we want.
    a, b = cos / scale, sin / scale
    d, e = -sin / scale, cos / scale
    c = cx - a * (cx + tx) - b * (cy + ty)
    f = cy - d * (cx + tx) - e * (cy + ty)
    fill = tuple(int(v) for v in border_colour(img))
    out = Image.fromarray(img).transform((w, h), Image.Transform.AFFINE, (a, b, c, d, e, f),
                                         resample=Image.Resampling.BILINEAR, fillcolor=fill)
    return np.asarray(out)


def turn90(img, quarter_turns: int):
    """An exact quarter, half or three-quarter turn (no smoothing), for the stress test."""
    return np.ascontiguousarray(np.rot90(img, quarter_turns))


def flip(img, rng):
    """A mirror image. A part seen from the other side is still the same part."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------- LIGHTING

def lighting(img, rng, brightness=None, contrast=None, gamma=None, tint=None):
    """
    Brighter or darker, flatter or punchier, and a colour tint (warm bulbs,
    cold tube lights). The tint matters for one class in particular:
    contamination is the ONLY class with colour in training, so a model can
    cheat by learning "any colour = contamination", and a warm light would then
    turn every part into contamination. Tinting every class teaches it that
    colour across the whole picture means lighting, not dirt.
    """
    x = img.astype(np.float32) / 255.0
    brightness = rng.uniform(0.6, 1.35) if brightness is None else brightness
    contrast = rng.uniform(0.6, 1.4) if contrast is None else contrast
    gamma = rng.uniform(0.7, 1.5) if gamma is None else gamma
    tint = rng.uniform(0.85, 1.15, size=3) if tint is None else np.asarray(tint)
    mean = x.mean()
    x = (x - mean) * contrast + mean
    x = x * brightness * tint
    x = np.clip(x, 0, 1) ** gamma
    return (x * 255).clip(0, 255).astype(np.uint8)


def uneven_light(img, rng, strength=None, angle=None):
    """Light coming from one side: one edge of the picture brighter than the other."""
    h, w = img.shape[:2]
    strength = rng.uniform(0.15, 0.45) if strength is None else strength
    angle = rng.uniform(0, 2 * np.pi) if angle is None else angle
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = ((xx - w / 2) * np.cos(angle) + (yy - h / 2) * np.sin(angle)) / (w / 2)
    gain = 1 + strength * ramp
    return (img.astype(np.float32) * gain[..., None]).clip(0, 255).astype(np.uint8)


def hue_change(img, rng, amount=None):
    """
    Turn every colour round the colour wheel. Grey pixels stay grey, so this
    only changes things that HAVE colour — mainly contamination specks. It
    teaches "coloured specks of any colour = contamination", in case the test
    shows blue or red dirt where training only had brown and green.
    """
    amount = int(rng.integers(0, 256)) if amount is None else amount
    h, s, v = Image.fromarray(img).convert("HSV").split()
    h = h.point(lambda value: (value + amount) % 256)
    return np.asarray(Image.merge("HSV", (h, s, v)).convert("RGB"))


def greyscale(img, rng=None):
    """Colour removed entirely — as a black-and-white inspection camera would see it."""
    g = (img.astype(np.float32) @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


# ---------------------------------------------------------------- BACKGROUNDS

def new_background(img, rng, kind=None, mask=None):
    """
    Cut the part out and put it on a different background: dark, bright,
    coloured, a gradient, or a noisy texture. Every training background is a
    flat pale colour; the hidden test changes backgrounds.
    Returns the image unchanged when the part cannot be cut out safely.
    """
    mask = background_mask(img) if mask is None else mask
    if mask is None:
        return img
    h, w = img.shape[:2]
    kind = rng.choice(["dark", "light", "colour", "gradient", "texture"]) if kind is None else kind
    if kind == "dark":
        bg = np.full((h, w, 3), rng.uniform(10, 70, size=3))
    elif kind == "light":
        bg = np.full((h, w, 3), rng.uniform(200, 255, size=3))
    elif kind == "colour":
        bg = np.full((h, w, 3), rng.uniform(0, 255, size=3))
    elif kind == "gradient":
        a, b = rng.uniform(0, 255, size=3), rng.uniform(0, 255, size=3)
        t = np.linspace(0, 1, w)[None, :, None]
        bg = np.broadcast_to(a * (1 - t) + b * t, (h, w, 3))
        if rng.random() < 0.5:
            bg = bg.transpose(1, 0, 2)
    else:  # texture — like a workbench, cloth, or a conveyor belt
        raw = np.clip(rng.normal(128, 40, size=(h, w)), 0, 255).astype(np.uint8)
        smooth = np.asarray(Image.fromarray(raw).filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 3))))
        bg = rng.uniform(40, 220, size=3) + (smooth.astype(np.float32) - 128)[..., None]
    out = img.copy()
    out[mask] = np.clip(bg, 0, 255).astype(np.uint8)[mask]
    return out


# ---------------------------------------------------------------- OCCLUSION / GLARE

def patches(img, rng, count=None, size=None):
    """
    Paste pale grey-to-white rectangles over the picture: glare, a sticker,
    a reflection, something in the way. The VALIDATION images already contain
    exactly this (the training images do not) — that is how we know it is one
    of the organisers' traps.

    It matters most for missing_component, whose defect is "one dot is white".
    A model that learned "white blob = missing component" would reject good
    parts that merely have glare on them.
    """
    h, w = img.shape[:2]
    out = img.copy()
    count = int(rng.integers(1, 4)) if count is None else count
    for _ in range(count):
        s = int(rng.integers(5, 24)) if size is None else size
        ph, pw = s, max(3, int(s * rng.uniform(0.7, 1.4)))
        y, x = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
        shade = rng.uniform(170, 255)
        alpha = rng.uniform(0.7, 1.0)
        region = out[y:y + ph, x:x + pw].astype(np.float32)
        out[y:y + ph, x:x + pw] = (region * (1 - alpha) + shade * alpha).astype(np.uint8)
    return out


# ---------------------------------------------------------------- CAMERA QUALITY

def blur(img, rng, sigma=None):
    """Out of focus, or the part moving on the belt."""
    sigma = rng.uniform(0.3, 1.8) if sigma is None else sigma
    return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(sigma)))


def noise(img, rng, std=None):
    """Grainy picture — a cheap camera or a dim room."""
    std = rng.uniform(3, 14) if std is None else std
    return (img.astype(np.float32) + rng.normal(0, std, img.shape)).clip(0, 255).astype(np.uint8)


def speckle(img, rng, amount=None):
    """
    Dead and stuck pixels: single black and white dots. Added after the stress
    test showed these made the model call almost everything contamination —
    scattered dark pixels look like dirt specks unless it has seen the difference.
    """
    amount = rng.uniform(0.005, 0.04) if amount is None else amount
    out = img.copy()
    dice = rng.random(img.shape[:2])
    out[dice < amount / 2] = 0
    out[dice > 1 - amount / 2] = 255
    return out


def sharpen(img, rng, percent=None):
    """
    A camera that over-sharpens, drawing bright and dark halos round every edge.
    Added after the stress test: a halo round a component dot looks like the
    outlined circle of a dent unless the model has seen the difference.
    """
    percent = int(rng.integers(80, 260)) if percent is None else percent
    return np.asarray(Image.fromarray(img).filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=0)))


def low_quality(img, rng, quality=None):
    """Heavy JPEG compression — pictures squeezed small to send over a network."""
    quality = int(rng.integers(25, 80)) if quality is None else quality
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


# ---------------------------------------------------------------- the whole simulator

# Each family can be switched off by name. That is how we measure what each one
# is worth: train without it, then test on exactly that change (evaluate.py).
FAMILIES = ("orientation", "lighting", "background", "occlusion", "camera")


# AUGMENTATION — every training photo passes through simulate(): orientation, lighting, background, glare, camera
def simulate(img, rng, families=FAMILIES, mask=None):
    """
    The training-time disturbance. Each family fires with some probability, so
    the model sees a mix of clean and disturbed parts, never the same copy twice.
    Background goes first because cutting the part out works best on the
    original picture, before anything else has changed it. `mask` is the
    background mask worked out in advance for this image (it never changes, so
    computing it once saves time every epoch).
    """
    if "background" in families and rng.random() < 0.5:
        img = new_background(img, rng, mask=mask)
    if "orientation" in families:
        img = rotate(img, rng)
        img = flip(img, rng)
    if "lighting" in families:
        if rng.random() < 0.8:
            img = lighting(img, rng)
        if rng.random() < 0.3:
            img = uneven_light(img, rng)
        if rng.random() < 0.3:
            img = hue_change(img, rng)
        if rng.random() < 0.1:
            img = greyscale(img)
    if "occlusion" in families and rng.random() < 0.5:
        img = patches(img, rng)
    if "camera" in families:
        if rng.random() < 0.3:
            img = blur(img, rng)
        if rng.random() < 0.15:
            img = sharpen(img, rng)
        if rng.random() < 0.3:
            img = noise(img, rng)
        if rng.random() < 0.15:
            img = speckle(img, rng)
        if rng.random() < 0.2:
            img = low_quality(img, rng)
    return img
