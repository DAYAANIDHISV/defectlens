"""
inference.py — asking a trained model for answers.

Used by evaluate.py, predict.py and the demo app, so all three answer the
same question in the same way.

TEST-TIME AUGMENTATION (TTA): instead of showing the model a photo once, we
show it eight ways — turned 0/90/180/270 degrees, each also mirrored — and
average what it says. A defect is a defect whichever way up the part is, so
eight views give a steadier answer than one, especially on the hidden test's
new orientations.

CALIBRATION (temperature): the raw scores are divided by one number, T,
before being turned into probabilities. T > 1 makes the model less sure of
itself. We choose T so that "90% confident" really means right about 90% of
the time (evaluate.py works it out). The best answer never changes — only how
sure we say we are.
"""
import numpy as np
import torch

EIGHT_VIEWS = [(turns, mirror) for turns in range(4) for mirror in (False, True)]


def to_batch(images) -> torch.Tensor:
    """A list of 128x128x3 photos (numbers 0-255) -> one tensor the model accepts (numbers 0-1)."""
    # .contiguous() lays the numbers out in memory in the new order; the Apple GPU's
    # gradient code (used for the heat-maps) refuses a batch that is only "viewed" reordered.
    batch = torch.from_numpy(np.stack([np.asarray(im) for im in images])).permute(0, 3, 1, 2)
    return batch.contiguous().float() / 255.0


# TEST-TIME AUGMENTATION — every photo seen 8 ways (turned + mirrored), each also zoomed; answers averaged
@torch.no_grad()
def raw_scores(model, images, tta: bool = True, batch: int = 128, zooms=(1.0,)) -> np.ndarray:
    """
    The model's six raw scores per photo, averaged over the eight views when tta
    is on. `zooms` adds magnified views too: 1.3 means "also look at the middle
    of the photo enlarged 1.3 times", which makes a part photographed from far
    away look closer. (The model resizes whatever it is given, so a zoom is just
    a crop of the centre.)
    """
    model.eval()
    device = next(model.parameters()).device
    views = EIGHT_VIEWS if tta else [(0, False)]
    out = []
    for start in range(0, len(images), batch):
        x = to_batch(images[start:start + batch]).to(device)
        total, count = 0, 0
        for zoom in zooms:
            size = x.shape[-1]
            keep = int(round(size / zoom))
            edge = (size - keep) // 2
            zx = x[:, :, edge:edge + keep, edge:edge + keep]
            for turns, mirror in views:
                v = torch.rot90(zx, turns, dims=(2, 3))
                if mirror:
                    v = torch.flip(v, dims=(3,))
                total = total + model(v.contiguous())
                count += 1
        out.append((total / count).cpu())
    return torch.cat(out).numpy()


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Scores -> probabilities that add up to 1 (the bigger a score, the bigger its share)."""
    z = scores / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    The T that makes the confidences honest on photos the model did not learn
    from. We try a range of values and keep the one with the lowest "surprise"
    (log loss): the average of -log(probability given to the true class).
    """
    best_t, best_loss = 1.0, float("inf")
    for t in np.arange(0.3, 5.0, 0.05):
        p = softmax(scores, t)[np.arange(len(labels)), labels]
        loss = float(-np.log(np.clip(p, 1e-12, 1)).mean())
        if loss < best_loss:
            best_t, best_loss = float(t), loss
    return best_t


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """
    How far confidence is from reality, on average. Group answers by how sure
    the model was (0.9-1.0, 0.8-0.9, ...); in each group compare the average
    confidence with the share actually right. 0 = perfectly honest.
    """
    confidence = probs.max(axis=1)
    right = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    gap = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (confidence > lo) & (confidence <= hi)
        if inside.any():
            gap += inside.mean() * abs(confidence[inside].mean() - right[inside].mean())
    return float(gap)
