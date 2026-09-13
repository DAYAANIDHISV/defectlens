"""
explain.py — WHERE did the model look? (Grad-CAM heat-maps)

A confident answer is not enough: a model can be right for the wrong reason
(the cow-on-grass problem). A heat-map shows which part of the photo pushed
the model towards its answer. On a scratch it should glow on the scratch; if it
glows on the background, the model is cheating.

How Grad-CAM works, in three steps:
  1. The deepest picture layer of ResNet18 (layer4) holds 512 small maps
     (7 x 7), each reacting to some pattern — a line here, a circle there.
  2. We ask: if this map got a little stronger, how much would the score for
     the chosen class go up? (That is the gradient.) Maps that raise the
     score get a big weight.
  3. Add the maps together with those weights, keep only the positive part,
     and stretch the 7 x 7 result to the size of the photo.
"""
import numpy as np
import torch
import torch.nn.functional as F

from inference import to_batch


# HEAT-MAP — Grad-CAM: where on the photo the model looked
def grad_cam(model, img, class_index=None):
    """A 128x128 map, 0 = ignored, 1 = most important, for one photo and one class."""
    device = next(model.parameters()).device
    store = {}

    def keep(_, __, output):
        """Hook: store layer4's output and ask PyTorch to keep its gradient."""
        output.retain_grad()
        store["maps"] = output

    hook = model.backbone.layer4.register_forward_hook(keep)
    try:
        with torch.enable_grad():
            model.zero_grad()
            scores = model(to_batch([img]).to(device))
            chosen = int(scores.argmax(1)) if class_index is None else class_index
            scores[0, chosen].backward()
    finally:
        hook.remove()
    maps, grads = store["maps"], store["maps"].grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * maps).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def overlay(img, cam, strength=0.55):
    """Paint the heat-map over the photo: dark blue = ignored, yellow-red = where it looked."""
    c = cam[..., None]
    heat = np.concatenate([np.clip(1.6 * c, 0, 1),            # red rises first
                           np.clip(1.6 * c - 0.6, 0, 1),      # then green (making yellow)
                           np.clip(0.6 - 1.5 * c, 0, 1)], 2)  # blue only where it is cold
    alpha = strength * np.clip(c * 1.4, 0.15, 1)
    out = img.astype(np.float32) / 255 * (1 - alpha) + heat * alpha
    return (out * 255).clip(0, 255).astype(np.uint8)
