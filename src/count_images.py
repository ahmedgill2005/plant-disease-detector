from pathlib import Path

for dataset in ["plantvillage", "plantdoc"]:
    for split in ["train", "test"]:
        folder = Path("data") / dataset / split
        if not folder.exists():
            print(f"{dataset}/{split}: NOT FOUND")
            continue
        classes = sorted(p for p in folder.iterdir() if p.is_dir())
        n_images = sum(1 for c in classes for f in c.iterdir() if f.is_file())
        print(f"{dataset}/{split}: {len(classes)} classes, {n_images} images")

print("\nPlantDoc class names:")
for c in sorted(p.name for p in Path("data/plantdoc/train").iterdir() if p.is_dir()):
    print(" ", c)
