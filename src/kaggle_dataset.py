from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def download_kaggle_dataset(slug: str, output_dir: str | Path) -> str:
    """Download a public Kaggle dataset using KaggleHub and copy it into the project.

    Example slug: 'andrewmvd/face-mask-detection' or 'mrsamhri/helmet-datasets'
    """
    import kagglehub

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(kagglehub.dataset_download(slug))
    destination = output_dir / slug.rsplit("/", 1)[-1]
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Downloaded dataset files to: {destination}")
    return str(destination)


def main():
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset and prepare it for this project")
    parser.add_argument("--slug", type=str, required=True, help="Kaggle dataset slug, e.g. andrewmvd/face-mask-detection")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Directory to store downloaded data")
    args = parser.parse_args()

    download_kaggle_dataset(args.slug, args.output_dir)


if __name__ == "__main__":
    main()
