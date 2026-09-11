"""
model.py — the model itself.

We do not build a model from nothing. We take ResNet18, a well-known picture
model that already learned to see (edges, shapes, textures) from 1.2 million
everyday photos (the ImageNet collection), and replace its very last layer —
the one that used to choose between 1,000 everyday objects — with a new one
that chooses between our six classes. Training then adjusts the whole network
a little for our defects. This is called FINE-TUNING or TRANSFER LEARNING.

The model takes pictures as numbers from 0 to 1 at any size. It enlarges them
to 224 x 224 (the size ResNet18 learned on) and shifts the colours the way
ImageNet training expected, so no other file has to remember to do either.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from config import CLASSES, MODELS_DIR, MODEL_SIZE

# The average colour and spread of ImageNet photos. The pretrained layers expect
# their input centred this way, so we do the same.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PRETRAINED_WEIGHTS = MODELS_DIR / "resnet18-imagenet.pth"


class DefectNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = torchvision.models.resnet18()
        if pretrained:
            self.backbone.load_state_dict(torch.load(PRETRAINED_WEIGHTS, weights_only=True))
        # The new last layer: 512 numbers describing the picture -> 6 scores, one per class.
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(CLASSES))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        """x: a batch of pictures, shape (count, 3, height, width), values 0-1. Returns 6 raw scores each."""
        x = F.interpolate(x, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear", align_corners=False)
        return self.backbone((x - self.mean) / self.std)


def pick_device() -> torch.device:
    """Apple's graphics chip (MPS) when there is one — about ten times faster than the CPU for this."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save(model: DefectNet, path, **info):
    torch.save({"state_dict": model.state_dict(), "info": info}, path)


def load(path, device=None) -> DefectNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = DefectNet(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.info = checkpoint.get("info", {})
    return model.to(device or pick_device()).eval()
