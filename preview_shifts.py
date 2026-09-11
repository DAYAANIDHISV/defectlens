"""
preview_shifts.py — a picture of what the shift simulator does.

One part from each class, then the same part under each disturbance family,
then three fully random training copies. Saved to docs/shift_examples.png.
Run it after any change to shifts.py: a bug in the disturbances ruins
training silently, and a picture catches it in seconds.
"""
import numpy as np
from PIL import Image, ImageDraw

import shifts
from config import CLASSES, ROOT
from data import load_train

COLUMNS = [
    ("original", lambda im, r: im),
    ("turned", lambda im, r: shifts.flip(shifts.rotate(im, r), r)),
    ("lighting", lambda im, r: shifts.lighting(im, r)),
    ("side light", lambda im, r: shifts.uneven_light(im, r, strength=0.45)),
    ("colours turned", lambda im, r: shifts.hue_change(im, r, amount=90)),
    ("background", lambda im, r: shifts.new_background(im, r)),
    ("glare", lambda im, r: shifts.patches(im, r)),
    ("camera", lambda im, r: shifts.low_quality(shifts.noise(shifts.blur(im, r), r), r)),
    ("random 1", lambda im, r: shifts.simulate(im, r)),
    ("random 2", lambda im, r: shifts.simulate(im, r)),
    ("random 3", lambda im, r: shifts.simulate(im, r)),
]


def main():
    images, labels, _ = load_train()
    rng = np.random.default_rng(7)
    size, pad = 128, 16
    sheet = Image.new("RGB", (len(COLUMNS) * size, len(CLASSES) * (size + pad) + pad), "white")
    draw = ImageDraw.Draw(sheet)
    for c, (title, _) in enumerate(COLUMNS):
        draw.text((c * size + 4, 2), title, fill="black")
    for row, name in enumerate(CLASSES):
        img = images[int(np.flatnonzero(labels == row)[3])]
        top = pad + row * (size + pad)
        draw.text((4, top - 1), name, fill="black")
        for c, (_, fn) in enumerate(COLUMNS):
            sheet.paste(Image.fromarray(fn(img, rng)), (c * size, top + pad // 2))
    out = ROOT / "docs" / "shift_examples.png"
    sheet.save(out)
    masks = [shifts.background_mask(im) for im in images]
    print(f"saved {out}")
    print(f"background could be cut out safely on {sum(m is not None for m in masks)} of {len(images)} training images")


if __name__ == "__main__":
    main()
