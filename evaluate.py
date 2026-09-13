"""
evaluate.py — the STRESS TEST, error analysis and calibration.

    python evaluate.py                        # score every trained model under every condition
    python evaluate.py --detail full          # plus a deep look at one model: mix-ups, a
                                              # gallery of its mistakes, and its calibration

The validation photos are easy (a model gets 0.97 after two epochs), so on
their own they tell us almost nothing about the hidden test, which changes
"lighting, orientation, backgrounds and defect presentation". So we re-draw
every validation photo under each condition below and score each condition
separately. Every model sees exactly the same altered photos (fixed seeds).

Each condition is labelled with how it relates to training, because a test
the model has practised for is weaker evidence than a surprise:

  as given        the validation photos untouched
  inside range    a kind of change the shift simulator makes, at a strength it uses
  beyond range    a kind of change it makes, but stronger than training ever did
  never simulated a change the shift simulator never makes — the honest proxy
                  for the hidden test's surprises
"""
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import shifts
from config import CLASSES, MODELS_DIR, OUTPUTS_DIR, ROOT, SEED
from data import load_validation
from inference import expected_calibration_error, fit_temperature, raw_scores, softmax
from metrics import macro_f1, report
from model import load


def far_away(img, rng):
    """The part at 65% of its size, as if photographed from further away."""
    return shifts.rotate(img, rng, angle=0, scale=0.65, shift=(0, 0))


def low_resolution(img, rng):
    """A cheaper camera: shrunk to 40 x 40 pixels and blown back up."""
    small = Image.fromarray(img).resize((40, 40), Image.Resampling.BILINEAR)
    return np.asarray(small.resize((128, 128), Image.Resampling.BILINEAR))


def tilted_camera(img, rng, squeeze=22):
    """The camera looking at the part from an angle: the far edge looks narrower than the near one."""
    w = h = img.shape[0]
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(squeeze, 0), (w - squeeze, 0), (w, h), (0, h)]
    rows, rhs = [], []
    for (x, y), (u, v) in zip(dst, src):   # Pillow wants output -> input, as for rotate()
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]]
        rhs += [u, v]
    coeffs = np.linalg.solve(np.array(rows, float), np.array(rhs, float))
    fill = tuple(int(c) for c in shifts.border_colour(img))
    return np.asarray(Image.fromarray(img).transform((w, h), Image.Transform.PERSPECTIVE, tuple(coeffs),
                                                     resample=Image.Resampling.BILINEAR, fillcolor=fill))


def motion_blur(img, rng, length=7):
    """A streak sideways — the part moving on the belt while the shutter is open."""
    streak = sum(np.roll(img.astype(np.float32), k - length // 2, axis=1) for k in range(length))
    return (streak / length).astype(np.uint8)


def shadow(img, rng):
    """A soft round shadow over one corner of the part (an arm, a tool, the camera itself)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    distance = np.hypot(xx - w * 0.3, yy - h * 0.35) / (w * 0.45)
    gain = 1 - 0.5 * np.clip(1.2 - distance, 0, 1)
    return (img * gain[..., None]).clip(0, 255).astype(np.uint8)


def salt_and_pepper(img, rng, amount=0.03):
    """Dead and stuck pixels: single black and white dots scattered everywhere."""
    out = img.copy()
    dice = rng.random(img.shape[:2])
    out[dice < amount / 2] = 0
    out[dice > 1 - amount / 2] = 255
    return out


def sharpened(img, rng):
    """A camera that over-sharpens: edges get bright and dark halos."""
    return np.asarray(Image.fromarray(img).filter(ImageFilter.UnsharpMask(radius=2, percent=250, threshold=0)))


def everything(img, rng):
    """Several changes at once: dark background, turned 90°, warm light, glare and blur."""
    img = shifts.new_background(img, rng, kind="dark")
    img = shifts.turn90(img, 1)
    img = shifts.lighting(img, rng, brightness=0.9, contrast=1.0, gamma=1.0, tint=(1.15, 1.0, 0.8))
    img = shifts.patches(img, rng, count=2)
    return shifts.blur(img, rng, sigma=1.2)


# name, relationship to training, how to make it
# STRESS TEST — the 29 changed conditions, each scored separately
CONDITIONS = [
    ("as given", "as given", lambda im, r: im),
    ("turned 90°", "inside range", lambda im, r: shifts.turn90(im, 1)),
    ("turned 180°", "inside range", lambda im, r: shifts.turn90(im, 2)),
    ("any angle", "inside range", lambda im, r: shifts.rotate(im, r)),
    ("mirrored", "inside range", lambda im, r: np.ascontiguousarray(im[:, ::-1])),
    ("dim light", "beyond range", lambda im, r: shifts.lighting(im, r, 0.5, 0.8, 1.0, (1, 1, 1))),
    ("harsh light", "beyond range", lambda im, r: shifts.lighting(im, r, 1.45, 1.5, 0.8, (1, 1, 1))),
    ("warm light", "beyond range", lambda im, r: shifts.lighting(im, r, 1.0, 1.0, 1.0, (1.2, 1.0, 0.75))),
    ("cool light", "beyond range", lambda im, r: shifts.lighting(im, r, 1.0, 1.0, 1.0, (0.8, 0.95, 1.2))),
    ("side light", "beyond range", lambda im, r: shifts.uneven_light(im, r, strength=0.6)),
    ("black & white", "inside range", lambda im, r: shifts.greyscale(im)),
    ("colours turned", "inside range", lambda im, r: shifts.hue_change(im, r, amount=128)),
    ("dark background", "inside range", lambda im, r: shifts.new_background(im, r, kind="dark")),
    ("textured background", "inside range", lambda im, r: shifts.new_background(im, r, kind="texture")),
    ("coloured background", "inside range", lambda im, r: shifts.new_background(im, r, kind="colour")),
    ("heavy glare", "beyond range", lambda im, r: shifts.patches(im, r, count=4)),
    ("blur", "beyond range", lambda im, r: shifts.blur(im, r, sigma=2.5)),
    ("grain", "beyond range", lambda im, r: shifts.noise(im, r, std=20)),
    ("squeezed JPEG", "beyond range", lambda im, r: shifts.low_quality(im, r, quality=15)),
    # "far away" was never simulated in the first recipe, and is how we found the
    # small-spots bug. The zoom range now covers it, so it counts as practised.
    ("far away", "inside range", far_away),
    ("low resolution", "never simulated", low_resolution),
    ("everything at once", "beyond range", everything),
    ("very far", "beyond range", lambda im, r: shifts.rotate(im, r, angle=0, scale=0.5, shift=(0, 0))),
    ("close-up", "beyond range", lambda im, r: shifts.rotate(im, r, angle=0, scale=1.35, shift=(0, 0))),
    ("off-centre", "beyond range", lambda im, r: shifts.rotate(im, r, angle=0, scale=1.0, shift=(22, -18))),
    ("tilted camera", "never simulated", tilted_camera),
    ("motion blur", "never simulated", motion_blur),
    ("shadow", "never simulated", shadow),
    # These two were never simulated when the stress test found them breaking
    # every model (bug 2). The camera family now makes both, so they count as practised.
    ("salt & pepper", "inside range", salt_and_pepper),
    ("sharpened", "inside range", sharpened),
]


def build_stress_sets(images):
    """Every validation photo under every condition, made once with fixed dice."""
    sets = {}
    for k, (name, _, fn) in enumerate(CONDITIONS):
        rng = np.random.default_rng(SEED + k)
        sets[name] = [fn(im, rng) for im in images]
    return sets


def error_gallery(images, truth, predicted, confidence, where, path, limit=48):
    """A picture of the model's mistakes, each captioned 'truth -> answer (confidence)'."""
    wrong = np.flatnonzero(truth != predicted)[:limit]
    if len(wrong) == 0:
        return 0
    cols, size, cap = 8, 128, 30
    rows = int(np.ceil(len(wrong) / cols))
    sheet = Image.new("RGB", (cols * size, rows * (size + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    for n, i in enumerate(wrong):
        x, y = (n % cols) * size, (n // cols) * (size + cap)
        sheet.paste(Image.fromarray(images[i]), (x, y + cap))
        draw.text((x + 2, y + 1), f"{CLASSES[truth[i]][:13]} -> {CLASSES[predicted[i]][:13]}", fill="black")
        draw.text((x + 2, y + 14), f"{confidence[i]:.2f} · {where[i][:18]}", fill=(120, 0, 0))
    sheet.save(path)
    return len(wrong)


def main():
    """Score every model under every condition, print a summary line each, write the table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="model names in models/ (default: all)")
    parser.add_argument("--detail", help="one model to analyse in depth")
    parser.add_argument("--tta", action="store_true", help="use the eight-view averaging for the table too")
    parser.add_argument("--zooms", type=float, nargs="+", default=[1.0],
                        help="also average magnified views, e.g. --zooms 1.0 1.3")
    parser.add_argument("--out", default="", help="suffix for the output table name")
    args = parser.parse_args()

    images, labels, _ = load_validation()
    stress = build_stress_sets(images)
    names = args.models or sorted(p.stem for p in MODELS_DIR.glob("*.pt"))
    OUTPUTS_DIR.mkdir(exist_ok=True)

    table = {}
    for name in names:
        # "a+b+c" is an ENSEMBLE: the three models' probabilities are averaged.
        members = [load(MODELS_DIR / f"{m}.pt") for m in name.split("+")]

        def answers(images):
            """The answer for each photo: probabilities averaged over the member models."""
            return sum(softmax(raw_scores(m, images, tta=args.tta, zooms=args.zooms)) for m in members).argmax(1)

        table[name] = {c: macro_f1(labels, answers(stress[c])) for c, _, _ in CONDITIONS}
        stressed = [c for c, kind, _ in CONDITIONS if kind != "as given"]
        print(f"{name:>18}: as given {table[name]['as given']:.3f} | "
              f"mean under stress {np.mean([table[name][c] for c in stressed]):.3f} | "
              f"worst {min(table[name][c] for c in stressed):.3f}", flush=True)

    # The table, as markdown for the report.
    header = "| condition | kind | " + " | ".join(names) + " |"
    lines = [header, "|" + "---|" * (len(names) + 2)]
    for c, kind, _ in CONDITIONS:
        cells = [table[n][c] for n in names]
        best = max(cells)
        lines.append(f"| {c} | {kind} | " + " | ".join(f"**{v:.3f}**" if v == best else f"{v:.3f}" for v in cells) + " |")
    for label, keep in [("MEAN, all stress", lambda k: k != "as given"),
                        ("MEAN, beyond range", lambda k: k == "beyond range"),
                        ("MEAN, never simulated", lambda k: k == "never simulated")]:
        cells = [np.mean([table[n][c] for c, kind, _ in CONDITIONS if keep(kind)]) for n in names]
        lines.append(f"| **{label}** | | " + " | ".join(f"**{v:.3f}**" for v in cells) + " |")
    suffix = ("_tta" if args.tta else "") + (f"_{args.out}" if args.out else "")
    (OUTPUTS_DIR / f"stress_table{suffix}.md").write_text("\n".join(lines) + "\n")
    (OUTPUTS_DIR / f"stress_results{suffix}.json").write_text(json.dumps(table, indent=1))
    print(f"\nwrote outputs/stress_table{suffix}.md")

    if args.detail:
        detail(args.detail, images, labels, stress)


# ERROR ANALYSIS + CALIBRATION — confusion matrix, gallery of mistakes, temperature
def detail(name, images, labels, stress):
    """Error analysis and calibration for one model, with the eight-view averaging on."""
    model = load(MODELS_DIR / f"{name}.pt")
    all_images, all_labels, where = [], [], []
    for c, _, _ in CONDITIONS:
        all_images += stress[c]
        all_labels += list(labels)
        where += [c] * len(labels)
    all_labels = np.array(all_labels)

    plain_scores = raw_scores(model, all_images, tta=False)
    scores = raw_scores(model, all_images, tta=True)
    t = fit_temperature(scores, all_labels)
    before, after = softmax(scores), softmax(scores, t)
    predicted = scores.argmax(1)

    text = [f"MODEL {name}", "",
            f"one view : macro-F1 over all {len(all_labels)} stressed photos = {macro_f1(all_labels, plain_scores.argmax(1)):.3f}",
            f"8 views  : macro-F1 over all {len(all_labels)} stressed photos = {macro_f1(all_labels, predicted):.3f}", "",
            "== validation as given (8 views) ==", report(labels, predicted[:len(labels)]), "",
            "== every stress condition pooled (8 views) ==", report(all_labels, predicted), "",
            f"calibration: temperature T = {t:.2f}",
            f"  expected calibration error before {expected_calibration_error(before, all_labels):.3f} "
            f"-> after {expected_calibration_error(after, all_labels):.3f}   (0 = confidence is perfectly honest)"]
    text = "\n".join(text)
    print("\n" + text)
    (OUTPUTS_DIR / f"detail_{name}.txt").write_text(text + "\n")

    confidence = after.max(1)
    shown = error_gallery(all_images, all_labels, predicted, confidence, where, ROOT / "docs" / f"errors_{name}.png")
    print(f"\n{shown} mistakes drawn in docs/errors_{name}.png")

    calibration = MODELS_DIR / "calibration.json"
    saved = json.loads(calibration.read_text()) if calibration.exists() else {}
    saved[name] = t
    calibration.write_text(json.dumps(saved, indent=1))


if __name__ == "__main__":
    main()
