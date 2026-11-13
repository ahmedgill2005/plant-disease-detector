from huggingface_hub import HfApi

REPO = "mohanty/PlantVillage"

print("Files in the dataset:")
files = HfApi().list_repo_files(REPO, repo_type="dataset")
for f in files[:60]:
    print("  ", f)
print("Total files:", len(files))

print("\nTrying to load with the 'default' setting...")
try:
    from datasets import load_dataset

    ds = load_dataset(REPO, name="default")
    print(ds)
    split = next(iter(ds.values()))
    print("Columns and types:", split.features)
    print(
        "First example (without image):",
        {k: v for k, v in split[0].items() if not hasattr(v, "size")},
    )
except Exception as e:
    print("Loading failed:", type(e).__name__, e)
