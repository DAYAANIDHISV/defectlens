"""
train.py — teach the model.

    python train.py --name full                      # our recipe: every disturbance family on
    python train.py --name plain --plain             # the naive baseline: no disturbances at all
    python train.py --name no-background --without background   # leave one family out
    python train.py --name final --final             # train on train + validation, for the submission

What happens, in order:
  1. Load the photos (data.py).
  2. Each time a training photo is used, disturb it at random (shifts.py).
  3. Show the model a batch of 32, measure how wrong it was (the LOSS), and
     nudge every internal number slightly in the direction that makes it less
     wrong. That nudge is GRADIENT DESCENT; the optimiser that does it is AdamW.
  4. After every pass through all the photos (an EPOCH), score the untouched
     validation photos with macro-F1, so we can watch it learn.
  5. Save the model and a log of every epoch to models/ and outputs/.

We keep the LAST epoch rather than the best-scoring one. Picking the best
validation epoch quietly tunes the model to the validation photos, and then
the validation score stops being an honest estimate.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import shifts
from config import CLASSES, MODELS_DIR, OUTPUTS_DIR, SEED
from data import load_train, load_validation
from metrics import macro_f1
from model import DefectNet, pick_device, save


class Parts(Dataset):
    """The photos plus labels, disturbed on the fly when `families` is not empty."""

    def __init__(self, images, labels, families=(), seed=SEED):
        self.images, self.labels, self.families = images, labels, tuple(families)
        # The background mask of an image never changes, so work it out once, not every epoch.
        needs_mask = "background" in self.families
        self.masks = [shifts.background_mask(im) if needs_mask else None for im in images]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.families:
            img = shifts.simulate(img, self.rng, self.families, mask=self.masks[i])
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return x, int(self.labels[i])


def give_each_worker_its_own_dice(worker_id):
    """Each loader worker gets a different random stream, or they would all disturb photos identically."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))


@torch.no_grad()
def predict_batchwise(model, images, device, batch=128):
    """The model's answer (a class number) for each photo, with no disturbance."""
    model.eval()
    answers = []
    for start in range(0, len(images), batch):
        x = torch.from_numpy(np.stack(images[start:start + batch])).permute(0, 3, 1, 2).float() / 255.0
        answers.append(model(x.to(device)).argmax(dim=1).cpu())
    return torch.cat(answers).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--plain", action="store_true", help="no disturbances at all (the naive baseline)")
    parser.add_argument("--without", nargs="*", default=[], choices=shifts.FAMILIES,
                        help="disturbance families to leave out")
    parser.add_argument("--final", action="store_true", help="train on train + validation together")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="a different seed = different random choices = a second opinion (for ensembles and error bars)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    families = () if args.plain else tuple(f for f in shifts.FAMILIES if f not in args.without)

    train_images, train_labels, _ = load_train()
    val_images, val_labels, _ = load_validation()
    if args.final:
        train_images = train_images + val_images
        train_labels = np.concatenate([train_labels, val_labels])

    loader = DataLoader(Parts(train_images, train_labels, families, args.seed), batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        worker_init_fn=give_each_worker_its_own_dice, drop_last=True)

    model = DefectNet(pretrained=True).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader)
    # Learning rate: warm up over the first 10% of training, then glide down to zero (a
    # "cosine" curve). Big steps early to learn fast, small steps late to settle.
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps,
                                                   pct_start=0.1, anneal_strategy="cos")
    # Label smoothing: aim for 90% sure instead of 100%. Stops the model becoming
    # over-confident, which matters because we report a confidence for every answer.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"[{args.name}] device={device} images={len(train_images)} families={families or 'none'} epochs={args.epochs}")
    log = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += loss.item() * len(y)
            seen += len(y)
        val_f1 = macro_f1(val_labels, predict_batchwise(model, val_images, device))
        log.append({"epoch": epoch, "train_loss": total / seen, "val_macro_f1": val_f1})
        print(f"  epoch {epoch:2d}  loss {total / seen:.3f}  validation macro-F1 {val_f1:.3f}"
              f"{'  (validation is inside training — not a fair score)' if args.final else ''}"
              f"  [{time.time() - started:.0f}s]", flush=True)

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{args.name}.pt"
    save(model, path, name=args.name, families=list(families), epochs=args.epochs, lr=args.lr,
         final=args.final, classes=CLASSES, seed=args.seed)
    (OUTPUTS_DIR / f"train_log_{args.name}.json").write_text(json.dumps(log, indent=1))
    print(f"[{args.name}] saved {path}  final validation macro-F1 {log[-1]['val_macro_f1']:.3f}"
          f"  in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
