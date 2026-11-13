import re
import zipfile
from pathlib import Path

ZIP = Path("data/plantdoc.zip")
OUT = Path("data/plantdoc_extracted")
BAD_CHARS = r'[<>:"|?*]'

saved = 0
with zipfile.ZipFile(ZIP) as z:
    for name in z.namelist():
        if name.endswith("/"):
            continue
        parts = name.split("/")
        if parts[0] not in ("train", "test") or len(parts) != 3:
            continue
        split, class_name, file_name = parts
        safe_name = re.sub(BAD_CHARS, "_", file_name)
        dest = OUT / split / class_name / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(z.read(name))
        saved += 1

print("Images saved:", saved)
