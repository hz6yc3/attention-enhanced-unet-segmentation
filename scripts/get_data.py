"""
Fetch the dataset from Kaggle into data/ (the repository does not redistribute it).

Requires the Kaggle API: `pip install kaggle` and a token at ~/.kaggle/kaggle.json
(Kaggle: Settings -> API -> Create New Token). Then:

    python scripts/get_data.py

Expected result:
    data/training/images, data/training/groundtruth,
    data/training/images_generated, data/training/groundtruth_generated,
    data/test_set_images/
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = "sanadalali/satellite-images-for-road-segmentation"
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    if (DATA / "training" / "images").exists():
        print(f"data already present at {DATA}")
        return
    tmp = Path(tempfile.mkdtemp(prefix="kaggle_"))
    print(f"downloading {DATASET} ...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(tmp), "--unzip"], check=True)
    except FileNotFoundError:
        sys.exit("kaggle CLI not found: pip install kaggle, then place your API token at ~/.kaggle/kaggle.json")
    # The archive layout is not guaranteed; locate the folders wherever they were unpacked.
    training = next((p for p in tmp.rglob("training") if (p / "images").is_dir()), None)
    test = next((p for p in tmp.rglob("test_set_images") if p.is_dir()), None)
    if training is None:
        sys.exit(f"could not find a 'training/images' folder inside the download at {tmp}")
    DATA.mkdir(exist_ok=True)
    shutil.move(str(training), str(DATA / "training"))
    if test is not None:
        shutil.move(str(test), str(DATA / "test_set_images"))
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ["images", "groundtruth", "images_generated", "groundtruth_generated"]:
        n = len(list((DATA / "training" / sub).glob("*.png"))) if (DATA / "training" / sub).exists() else 0
        print(f"  training/{sub}: {n} files")
    print(f"done: {DATA}")


if __name__ == "__main__":
    main()
