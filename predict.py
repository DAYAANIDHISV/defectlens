"""
predict.py — a folder of photos in, submission.csv out.

    python predict.py path/to/hidden_test_images
    python predict.py path/to/images --models final --out submission.csv

This is the file that runs on the hidden test the moment we have it. For each
photo: eight views averaged (test-time augmentation), confidences made honest
(calibration), and one line written in exactly the organisers' format:

    sample_id,predicted_class,confidence

The sample_id is the file name without its extension (DLHT_0000.png -> DLHT_0000).
Passing several models averages their probabilities (an ENSEMBLE).
"""
import argparse
import csv
import json
from collections import Counter

import numpy as np

from config import CLASSES, MODELS_DIR, ROOT
from data import load_folder
from inference import raw_scores, softmax
from model import load

REVIEW_BELOW = 0.60   # the demo sends anything less sure than this to a person

# Views averaged per photo: eight turns/mirrors, each also magnified 1.3x from the
# centre. The magnified views were added after the stress test showed parts at
# half size were our weakest case (0.87 -> 0.99), at a cost of 1-3 photos on
# off-centre and blurred parts. Chosen once, not searched.
ZOOMS = (1.0, 1.3)

# The models the submission uses — the ONE place this is decided; app.py reads it
# too. Several names = an ensemble (probabilities averaged).
# Three copies of the final recipe with different random seeds: across seeds the
# stress average is 0.990 +/- 0.002 but the half-size case swings 0.78-0.93, and
# the three together lift the worst stress condition from 0.965 to 0.980.
SUBMISSION_MODELS = ["final", "final-s1", "final-s2"]


def submission_models():
    """The submission's models that exist on disk (falls back to the train-only twin)."""
    present = [n for n in SUBMISSION_MODELS if (MODELS_DIR / f"{n}.pt").exists()]
    return present or ["full-v3"]


FINAL_TWIN = "full-v3"   # the same recipe as the final model, trained on the training photos only


def temperature_for(name: str) -> float:
    """
    The calibration number for a model. A final model (trained on everything)
    has no photos left to calibrate on, so it borrows the one from its twin
    trained on the training photos alone, which used the same recipe.

    We never go BELOW 1.0. Calibrating on our stress test said the model could
    be twice as sure of itself (T = 0.5). But the stress test is our own guess at
    the hidden test's changes, and letting our own guess make the model MORE
    confident is trusting ourselves too much. On a production line an
    over-confident wrong answer is the dangerous one — a bad part gets passed —
    so we only ever allow calibration to make the model more careful.
    """
    path = MODELS_DIR / "calibration.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    fitted = saved.get(name, saved.get(FINAL_TWIN, 1.0) if name.startswith("final") else 1.0)
    return max(fitted, 1.0)


def predict(images, model_names):
    """Probabilities (count x 6), averaged over the given models."""
    total = 0
    for name in model_names:
        model = load(MODELS_DIR / f"{name}.pt")
        total = total + softmax(raw_scores(model, images, tta=True, zooms=ZOOMS), temperature_for(name))
    return total / len(model_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--models", nargs="+", default=submission_models())
    parser.add_argument("--out", default=str(ROOT / "submission.csv"))
    args = parser.parse_args()

    images, ids = load_folder(args.folder)
    if not images:
        raise SystemExit(f"no images found in {args.folder}")
    probs = predict(images, args.models)
    answers, confidence = probs.argmax(1), probs.max(1)

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "predicted_class", "confidence"])
        for sample_id, answer, conf in sorted(zip(ids, answers, confidence)):
            writer.writerow([sample_id, CLASSES[answer], f"{conf:.4f}"])

    counts = Counter(CLASSES[a] for a in answers)
    print(f"wrote {args.out}: {len(ids)} photos, models {args.models}, "
          f"temperatures {[round(temperature_for(m), 2) for m in args.models]}")
    for name in CLASSES:
        print(f"  {name:<18} {counts.get(name, 0)}")
    unsure = int((confidence < REVIEW_BELOW).sum())
    print(f"  {unsure} photos below {REVIEW_BELOW:.0%} confidence (the demo would send these to a person)")


if __name__ == "__main__":
    main()
