"""
show_heatmaps.py — is the model right for the right reason?

For two validation photos per class, draws the photo and its Grad-CAM heat-map
(explain.py) side by side, captioned with the model's answer. Saved to
docs/heatmaps_<model>.png.

    python show_heatmaps.py full
"""
import sys

import numpy as np
from PIL import Image, ImageDraw

from config import CLASSES, MODELS_DIR, ROOT
from data import load_validation
from explain import grad_cam, overlay
from inference import raw_scores, softmax
from model import load


def main(name="full"):
    """Draw photo + heat-map pairs, two per class, for the named model."""
    model = load(MODELS_DIR / f"{name}.pt")
    images, labels, _ = load_validation()
    size, cap = 128, 16
    sheet = Image.new("RGB", (4 * size, len(CLASSES) * (size + cap)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, cls in enumerate(CLASSES):
        for k, i in enumerate(np.flatnonzero(labels == row)[[1, 7]]):
            img = images[i]
            p = softmax(raw_scores(model, [img], tta=False))[0]
            x, y = k * 2 * size, row * (size + cap)
            sheet.paste(Image.fromarray(img), (x, y + cap))
            sheet.paste(Image.fromarray(overlay(img, grad_cam(model, img))), (x + size, y + cap))
            draw.text((x + 2, y + 2), f"{cls} -> {CLASSES[p.argmax()]} {p.max():.2f}", fill="black")
    out = ROOT / "docs" / f"heatmaps_{name}.png"
    sheet.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
