import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO = "mohanty/PlantVillage"
OUT = Path("data/plantvillage")


def read_split(name):
    path = hf_hub_download(REPO, f"splits/{name}.txt", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def main():
    splits = {"train": read_split("color_train"), "test": read_split("color_test")}
    for split, paths in splits.items():
        print(f"{split}: {len(paths)} colour images listed")

    print("\nDownloading data.zip (large file, please wait)...")
    zip_path = hf_hub_download(REPO, "data.zip", repo_type="dataset")

    with zipfile.ZipFile(zip_path) as z:
        index = {}
        for name in z.namelist():
            pos = name.find("raw/color/")
            if pos != -1 and not name.endswith("/"):
                index[name[pos:]] = name
        print("Colour images found inside the zip:", len(index))

        for split, paths in splits.items():
            missing = 0
            for rel_path in tqdm(paths, desc=f"Copying {split}"):
                member = index.get(rel_path)
                if member is None:
                    missing += 1
                    continue
                class_name = rel_path.split("/")[2]
                dest = OUT / split / class_name / Path(rel_path).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            print(f"{split}: done, {missing} listed images not found in zip")

    print("\nFinished. Images saved in", OUT.resolve())


if __name__ == "__main__":
    main()
