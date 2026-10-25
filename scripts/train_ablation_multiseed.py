import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ablation_model import UAFloodNet, training_loss
from data import NPZFloodDataset
from train_v4 import augment_misregistration


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0

    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                batch["sar_pre"], batch["sar_post"],
                batch["ms_pre"], batch["ms_post"]
            )["logits"]

            pred = logits.sigmoid() >= 0.5
            gt = batch["mask"] >= 0.5

            tp += (pred & gt).sum().item()
            fp += (pred & ~gt).sum().item()
            fn += (~pred & gt).sum().item()

    return {
        "flood_iou": tp / max(tp + fp + fn, 1),
        "flood_f1": 2 * tp / max(2 * tp + fp + fn, 1)
    }


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--width", type=int, default=24)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--shift-prob", type=float, default=0.0)
    p.add_argument("--max-shift", type=int, default=2)
    p.add_argument(
        "--variant",
        choices=["full", "no_alignment",
                 "no_reliability", "base"],
        required=True
    )

    args = p.parse_args()

    if not 0 <= args.shift_prob <= 1:
        raise ValueError("Invalid shift probability")

    if args.epochs < 1 or args.max_shift < 0:
        raise ValueError("Invalid epoch count or max shift")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    last_path = out / "last.pt"
    best_path = out / "best.pt"
    history_path = out / "history.json"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Separate generator makes shuffle state restorable.
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        NPZFloodDataset(args.train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        NPZFloodDataset(args.val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    use_alignment = args.variant not in (
        "no_alignment", "base"
    )
    use_reliability = args.variant not in (
        "no_reliability", "base"
    )

    model = UAFloodNet(
        width=args.width,
        use_alignment=use_alignment,
        use_reliability=use_reliability
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr
    )

    start_epoch = 1
    best_iou = -1.0
    history = []

    config = {
        "seed": args.seed,
        "variant": args.variant,
        "use_alignment": use_alignment,
        "use_reliability": use_reliability,
        "width": args.width,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "shift_prob": args.shift_prob,
        "max_shift": args.max_shift,
        "train": str(Path(args.train).resolve()),
        "val": str(Path(args.val).resolve()),
    }

    if last_path.exists():
        checkpoint = torch.load(
            last_path,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["config"] != config:
            raise RuntimeError(
                "Saved checkpoint configuration differs "
                "from the requested experiment."
            )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Transfer optimizer state tensors to training device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

        if "python_rng" in checkpoint:
            random.setstate(checkpoint["python_rng"])
        if "numpy_rng" in checkpoint:
            state = checkpoint["numpy_rng"]
            np.random.set_state((
                state[0],
                state[1].cpu().numpy()
                if torch.is_tensor(state[1]) else state[1],
                int(state[2]), int(state[3]), float(state[4])
            ))
        generator.set_state(checkpoint["shuffle_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng"]
            )

        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint["best_iou"]
        history = checkpoint["history"]

        print(
            f"Resuming after epoch {start_epoch-1}",
            flush=True
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            batch = augment_misregistration(
                batch,
                args.shift_prob,
                args.max_shift
            )

            optimizer.zero_grad(set_to_none=True)

            output = model(
                batch["sar_pre"], batch["sar_post"],
                batch["ms_pre"], batch["ms_post"]
            )

            loss, _ = training_loss(
                model, output, batch, 0.1, 0.1, 0.01
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0
            )

            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        mean_loss = total_loss / len(train_loader)

        record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            **metrics
        }
        history.append(record)

        improved = metrics["flood_iou"] > best_iou

        if improved:
            best_iou = metrics["flood_iou"]

            atomic_save({
                "model": model.state_dict(),
                "epoch": epoch,
                "width": args.width,
                "metrics": metrics,
                "seed": args.seed,
                "shift_prob": args.shift_prob,
                "max_shift": args.max_shift,
                "variant": args.variant,
                "use_alignment": use_alignment,
                "use_reliability": use_reliability
            }, best_path)

        # Save all state needed to continue training.
        atomic_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "history": history,
            "config": config,
            "python_rng": random.getstate(),
            "numpy_rng": (
                np.random.get_state()[0],
                torch.from_numpy(np.random.get_state()[1].copy()),
                np.random.get_state()[2],
                np.random.get_state()[3],
                np.random.get_state()[4]
            ),
            "shuffle_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" else []
            )
        }, last_path)

        history_path.write_text(
            json.dumps(history, indent=2)
        )

        print(
            f"epoch={epoch} "
            f"loss={mean_loss:.4f} "
            f"val_iou={metrics['flood_iou']:.4f} "
            f"val_f1={metrics['flood_f1']:.4f} "
            f"best_iou={best_iou:.4f}",
            flush=True
        )

    print(
        f"FINISHED epoch={args.epochs} "
        f"best_iou={best_iou:.4f}",
        flush=True
    )


if __name__ == "__main__":
    main()
