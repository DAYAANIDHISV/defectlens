"""
app.py — the DefectLens inspection station (our working prototype).

    .venv/bin/python app.py        then open  http://localhost:8050

What somebody at the station can do:
  * drop in a photo of a part (or click a sample) and get a verdict:
      PASS    — no defect, and the model is sure
      REJECT  — which defect it found
      REVIEW  — the model is not sure enough; a person should look
  * see how sure the model is, and its chance for each of the six classes
  * see a heat-map of WHERE on the part the model looked (explain.py)
  * press "stress it" buttons that change the photo — turn it, dim the light,
    swap the background, add glare, blur — and watch whether the answer holds.
    These are the same kinds of change the hidden test makes, so anyone can
    test our robustness claim live instead of taking our word for it.
  * drop in a whole folder and get submission.csv in the organisers' format.

The page (static/index.html) talks to this server through three small
addresses (the "API"): /api/inspect, /api/batch and /api/samples.
"""
import base64
import csv
import io

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import shifts
from config import CLASSES, MODELS_DIR, ROOT
from data import load_validation, standardise
from explain import grad_cam, overlay
from inference import raw_scores, softmax
from model import load
from predict import REVIEW_BELOW, ZOOMS, submission_models, temperature_for

# The same models the submission uses (decided in predict.py).
MODEL_NAMES = submission_models()
if not all((MODELS_DIR / f"{n}.pt").exists() for n in MODEL_NAMES):
    raise SystemExit("No trained models yet: train the three final models first (README, 'Run it on your own computer').")
MODELS = [(name, load(MODELS_DIR / f"{name}.pt"), temperature_for(name)) for name in MODEL_NAMES]

# The disturbances offered on the page. Same functions the stress test uses.
DISTURBANCES = {
    "turn 90°": lambda im, r: shifts.turn90(im, 1),
    "any angle": lambda im, r: shifts.rotate(im, r),
    "mirror": lambda im, r: np.ascontiguousarray(im[:, ::-1]),
    "dim light": lambda im, r: shifts.lighting(im, r, 0.55, 0.85, 1.0, (1, 1, 1)),
    "warm light": lambda im, r: shifts.lighting(im, r, 1.0, 1.0, 1.0, (1.2, 1.0, 0.75)),
    "side light": lambda im, r: shifts.uneven_light(im, r, strength=0.55),
    "dark background": lambda im, r: shifts.new_background(im, r, kind="dark"),
    "textured background": lambda im, r: shifts.new_background(im, r, kind="texture"),
    "glare": lambda im, r: shifts.patches(im, r, count=2),
    "blur": lambda im, r: shifts.blur(im, r, sigma=1.8),
    "grain": lambda im, r: shifts.noise(im, r, std=16),
    "black & white": lambda im, r: shifts.greyscale(im),
    "far away": lambda im, r: shifts.rotate(im, r, angle=0, scale=0.7, shift=(0, 0)),
}

app = Flask(__name__, static_folder=str(ROOT / "static"))
rng = np.random.default_rng()


def data_url(img: np.ndarray) -> str:
    """A picture as text the web page can show directly (PNG, base64-encoded)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def probabilities(images) -> np.ndarray:
    """Eight views (each also zoomed), calibrated, averaged over every loaded model."""
    return sum(softmax(raw_scores(m, images, tta=True, zooms=ZOOMS), t) for _, m, t in MODELS) / len(MODELS)


def verdict(label: str, confidence: float) -> str:
    if confidence < REVIEW_BELOW:
        return "REVIEW"
    return "PASS" if label == "normal" else "REJECT"


@app.get("/")
def page():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    return jsonify(models=[{"name": n, "temperature": round(t, 2)} for n, _, t in MODELS],
                   review_below=REVIEW_BELOW, classes=CLASSES, disturbances=list(DISTURBANCES))


@app.post("/api/inspect")
def inspect():
    """One photo (optionally disturbed first) -> verdict, probabilities, heat-map."""
    img = standardise(Image.open(request.files["image"].stream))
    disturbance = request.form.get("disturb")
    if disturbance:
        img = DISTURBANCES[disturbance](img, rng)
    p = probabilities([img])[0]
    best = int(p.argmax())
    cam = grad_cam(MODELS[0][1], img, class_index=best)
    return jsonify(image=data_url(img), heatmap=data_url(overlay(img, cam)),
                   label=CLASSES[best], confidence=float(p[best]), verdict=verdict(CLASSES[best], float(p[best])),
                   probabilities=[{"class": c, "p": float(v)} for c, v in zip(CLASSES, p)])


@app.post("/api/batch")
def batch():
    """Many photos -> one row each, plus submission.csv as text."""
    files = request.files.getlist("images")
    ids = [f.filename.rsplit("/", 1)[-1].rsplit(".", 1)[0] for f in files]
    images = [standardise(Image.open(f.stream)) for f in files]
    p = probabilities(images)
    rows = []
    for sample_id, img, probs in sorted(zip(ids, images, p), key=lambda r: r[0]):
        best = int(probs.argmax())
        rows.append({"id": sample_id, "label": CLASSES[best], "confidence": float(probs[best]),
                     "verdict": verdict(CLASSES[best], float(probs[best])), "thumb": data_url(img)})
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["sample_id", "predicted_class", "confidence"])
    for r in rows:
        writer.writerow([r["id"], r["label"], f"{r['confidence']:.4f}"])
    return jsonify(rows=rows, csv=out.getvalue())


@app.get("/api/samples")
def samples():
    """Two validation photos per class, for trying the station without files to hand."""
    images, labels, _ = load_validation()
    picks = []
    for k, name in enumerate(CLASSES):
        for i in np.flatnonzero(labels == k)[[2, 9]]:
            picks.append({"class": name, "image": data_url(images[i])})
    return jsonify(picks)


if __name__ == "__main__":
    print(f"DefectLens station: models {MODEL_NAMES}, open http://localhost:8050")
    app.run(host="127.0.0.1", port=8050, debug=False)
