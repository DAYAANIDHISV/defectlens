# DefectLens — robust visual defect inspection

AI ARENA 2026 · DefectLens track. Six classes: `normal`, `scratch`, `dent`, `contamination`,
`misalignment`, `missing_component`.

**Approach:** an ImageNet-pretrained ResNet18, fine-tuned on training photos that a *shift
simulator* disturbs at random (orientation, lighting, background, glare, camera quality), because
the private test changes exactly those. It is judged by a **29-condition stress test**, not by the
validation set alone, which turned out to be too easy to tell a robust model from a fragile one.

| macro-F1 | naive model | ours, one model | **ours, submitted** |
|---|---:|---:|---:|
| validation, as given | 0.985 | 1.000 | **1.000** |
| mean over 29 stress conditions | 0.849 | 0.990 ± 0.002 | **0.996** |
| worst stress condition | 0.292 | 0.78 – 0.93 | **0.980** |
| conditions never simulated in training (mean of 4) | 0.962 | 0.998 | **0.998** |

*One model* is a single view, averaged over three training seeds. *Submitted* is three seeds
averaged, each looking at 16 views of every photo.

- **Preprocessing:** any image → RGB, padded square, 128 × 128; the model upsamples to 224 and
  applies ImageNet normalisation.
- **Model:** ResNet18, all layers fine-tuned, 30 epochs, AdamW, label smoothing 0.1. Three copies
  with different random seeds, each trained on train + validation (950 images).
- **Prediction:** every photo seen 16 ways (4 rotations × mirror × 2 zooms), probabilities averaged
  over the three models. Confidence is that probability; calibration is only ever allowed to
  reduce it.
- **Limitations:** very small parts vary most between training runs; over-confidence when glare
  hides a component; synthetic data only. Details in `REPORT.md` §9.

## Run it

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
curl -L -o models/resnet18-imagenet.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
# put the organisers' data in data/DefectLens/{train,validation}/<class>/

for s in 42 1 2; do                                    # ~2.5 min each on an Apple M3 Pro
  .venv/bin/python train.py --final --seed $s --name $( [ $s = 42 ] && echo final || echo final-s$s )
done
.venv/bin/python predict.py path/to/test_images       # -> submission.csv
.venv/bin/python app.py                               # inspection station -> http://localhost:8050
```

## Files

| File | Purpose |
|---|---|
| `config.py` | paths, class order, sizes, seed |
| `data.py` | loading and `standardise()` — never reads the ID in a file name (it leaks the label) |
| `shifts.py` | the shift simulator |
| `model.py` | ResNet18 with a 6-way head |
| `train.py` | training, `--plain` baseline, `--without <family>` ablations, `--final` |
| `inference.py` | eight-view averaging, calibration |
| `metrics.py` | precision, recall, F1, macro-F1, confusion matrix — by hand |
| `evaluate.py` | the 29-condition stress test, error gallery, calibration |
| `explain.py` | Grad-CAM heat-maps |
| `predict.py` | folder → `submission.csv` |
| `app.py`, `static/index.html` | the inspection station |
| `preview_shifts.py`, `show_heatmaps.py` | the figures in `docs/` |
| `REPORT.md` | the technical report |
| `outputs/` | every stress table, the per-epoch training logs, the error analysis |
