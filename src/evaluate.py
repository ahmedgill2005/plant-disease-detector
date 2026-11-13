import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from class_map import PLANTDOC_TO_PLANTVILLAGE

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageList(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = timm.create_model(
        ckpt["model_name"], pretrained=False, num_classes=len(ckpt["class_names"])
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["class_names"], ckpt["img_size"]


@torch.no_grad()
def predict(model, items, img_size, device):
    tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    loader = DataLoader(ImageList(items, tf), batch_size=64, num_workers=4)
    all_probs = []
    for images, _ in loader:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")
        ):
            logits = model(images.to(device))
        all_probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(all_probs)


def report(name, y_true, probs, class_names, allowed=None):
    if allowed is not None:
        mask = np.full(probs.shape[1], -np.inf)
        mask[allowed] = 0
        probs = probs + mask
    pred = probs.argmax(axis=1)
    top5 = np.argsort(probs, axis=1)[:, -5:]
    top5_acc = np.mean([y in row for y, row in zip(y_true, top5)])
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, labels=np.unique(y_true), average="macro")
    print(
        f"{name:42s} | accuracy {acc*100:5.1f}% | top-5 {top5_acc*100:5.1f}% | macro-F1 {f1*100:5.1f}%"
    )
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/plantvillage_effb0.pt")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, img_size = load_model(args.model, device)
    print(f"Model: {args.model}  ({len(class_names)} classes)\n")

    pv_items = [
        (f, class_names.index(c.name))
        for c in sorted(Path("data/plantvillage/test").iterdir())
        for f in sorted(c.iterdir())
    ]
    pd_items = [
        (f, class_names.index(PLANTDOC_TO_PLANTVILLAGE[c.name]))
        for c in sorted(Path("data/plantdoc/test").iterdir())
        for f in sorted(c.iterdir())
    ]
    print(
        f"PlantVillage test images: {len(pv_items)},  PlantDoc test images: {len(pd_items)}\n"
    )

    pv_probs = predict(model, pv_items, img_size, device)
    pd_probs = predict(model, pd_items, img_size, device)

    pv_y = np.array([y for _, y in pv_items])
    pd_y = np.array([y for _, y in pd_items])
    allowed = sorted({class_names.index(v) for v in PLANTDOC_TO_PLANTVILLAGE.values()})

    report("PlantVillage test (lab photos)", pv_y, pv_probs, class_names)
    pd_pred = report(
        "PlantDoc test (real photos), all 38 classes", pd_y, pd_probs, class_names
    )
    report(
        "PlantDoc test (real photos), 28 classes", pd_y, pd_probs, class_names, allowed
    )

    out = Path("runs") / f"{Path(args.model).stem}_plantdoc_predictions.csv"
    pd.DataFrame(
        {
            "file": [str(f) for f, _ in pd_items],
            "true": [class_names[y] for y in pd_y],
            "predicted": [class_names[y] for y in pd_pred],
            "confidence": pd_probs.max(axis=1).round(3),
        }
    ).to_csv(out, index=False)
    print("\nSaved PlantDoc predictions to", out)


if __name__ == "__main__":
    main()
