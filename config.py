"""
config.py — settings every other script shares.

Where the data lives, the six class names, and the few numbers we tune.
Change something here and every script picks it up, so nothing gets
out of step with anything else.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "DefectLens"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# The six answers the model can give. The ORDER matters: the model outputs
# six numbers, and number 0 means "normal", number 1 means "scratch", and so on.
CLASSES = ["normal", "scratch", "dent", "contamination", "misalignment", "missing_component"]

IMAGE_SIZE = 128   # every image we were given is 128 x 128 pixels
MODEL_SIZE = 224   # we enlarge to 224 because the pretrained model learned on 224-pixel photos
SEED = 42          # fixes the "random" choices so a run can be repeated exactly
