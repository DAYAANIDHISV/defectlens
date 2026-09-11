"""
metrics.py — how we score answers. Written out by hand so every step can be explained.

CONFUSION MATRIX: a 6x6 table. Row = the true class, column = what the model said.
Right answers sit on the diagonal; everything off it is a specific mix-up.

For each class:
  PRECISION = of the parts we CALLED this class, the share that really were it.
  RECALL    = of the parts that REALLY WERE this class, the share we caught.
  F1        = 2 x P x R / (P + R). High only when both are high.

MACRO-F1 = the plain average of the six F1s. Every class counts equally, so
being useless at one class costs a sixth of the score however good the rest
are. This is how the organisers mark the hidden test.
"""
import numpy as np

from config import CLASSES


def confusion_matrix(true, pred, n: int = len(CLASSES)) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (np.asarray(true), np.asarray(pred)), 1)
    return matrix


def per_class(matrix: np.ndarray):
    """Precision, recall and F1 for each class, from a confusion matrix."""
    hits = np.diag(matrix).astype(float)
    called = matrix.sum(axis=0)      # column totals: how often the model said this class
    actual = matrix.sum(axis=1)      # row totals: how often it really was this class
    precision = np.divide(hits, called, out=np.zeros_like(hits), where=called > 0)
    recall = np.divide(hits, actual, out=np.zeros_like(hits), where=actual > 0)
    both = precision + recall
    f1 = np.divide(2 * precision * recall, both, out=np.zeros_like(hits), where=both > 0)
    return precision, recall, f1


def macro_f1(true, pred) -> float:
    return float(per_class(confusion_matrix(true, pred))[2].mean())


def accuracy(true, pred) -> float:
    return float((np.asarray(true) == np.asarray(pred)).mean())


def report(true, pred) -> str:
    """A readable table: per-class scores, then the confusion matrix."""
    matrix = confusion_matrix(true, pred)
    precision, recall, f1 = per_class(matrix)
    width = max(len(c) for c in CLASSES)
    lines = [f"{'class':<{width}}  precision  recall    f1"]
    for i, name in enumerate(CLASSES):
        lines.append(f"{name:<{width}}  {precision[i]:9.2f}  {recall[i]:6.2f}  {f1[i]:4.2f}")
    lines.append(f"{'MACRO-F1':<{width}}  {'':9}  {'':6}  {f1.mean():4.3f}   (accuracy {accuracy(true, pred):.3f})")
    lines.append("")
    short = [c[:5] for c in CLASSES]
    lines.append("confusion (rows = truth, columns = prediction)")
    lines.append(" " * (width + 2) + " ".join(f"{s:>5}" for s in short))
    for i, name in enumerate(CLASSES):
        lines.append(f"{name:<{width}}  " + " ".join(f"{v:5d}" for v in matrix[i]))
    return "\n".join(lines)
