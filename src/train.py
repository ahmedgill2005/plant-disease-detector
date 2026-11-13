import argparse
import csv
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/plantvillage/train")
    p.add_argument("--model", default="efficientnet_b0")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="models/plantvillage_effb0.pt")
    return p.parse_args()


def main():
    args = get_args()
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        "Using device:",
        device,
        torch.cuda.get_device_name(0) if device == "cuda" else "",
    )

    # ---------- Data ----------
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_ds = datasets.ImageFolder(args.data, transform=train_tf)
    class_names = train_ds.classes
    print(f"{len(train_ds)} training images, {len(class_names)} classes")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )

    # ---------- Model ----------
    model = timm.create_model(args.model, pretrained=True, num_classes=len(class_names))
    model = model.to(device).to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.1
    )

    # ---------- Log file ----------
    log_path = Path("runs") / f"{Path(args.out).stem}_log.csv"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "seconds"])

    # ---------- Training loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_loss, correct, seen = 0.0, 0, 0

        bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in bar:
            images = images.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)
            bar.set_postfix(loss=f"{total_loss/seen:.3f}", acc=f"{correct/seen:.3f}")

        secs = time.time() - start
        print(
            f"Epoch {epoch}: loss {total_loss/seen:.4f}, train accuracy {correct/seen:.4f}, {secs:.0f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    round(total_loss / seen, 4),
                    round(correct / seen, 4),
                    round(secs),
                ]
            )

    # ---------- Save ----------
    Path(args.out).parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_name": args.model,
            "class_names": class_names,
            "img_size": args.img_size,
            "state_dict": model.state_dict(),
        },
        args.out,
    )
    print("Saved model to", args.out)


if __name__ == "__main__":
    main()
