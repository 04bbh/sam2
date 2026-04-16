#!/usr/bin/env python3
"""Create missing track txt files by matching filenames from a reference dir."""

from __future__ import annotations

from pathlib import Path


def create_missing_track_txt(
    reference_dir: str | Path = "/data/users/wjq/codes/sam2-main/tracks/1",
    target_dir: str | Path = "/data/users/wjq/codes/sam2-main/tracks/4",
) -> list[Path]:
    """
    Match .txt filenames in reference_dir against target_dir.

    If target_dir is missing a .txt file that exists in reference_dir, create an
    empty .txt file with the same name in target_dir. Existing files are not
    modified.

    Returns the paths that were created.
    """
    reference_path = Path(reference_dir)
    target_path = Path(target_dir)

    if not reference_path.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {reference_path}")

    target_path.mkdir(parents=True, exist_ok=True)

    reference_names = {
        path.name for path in reference_path.iterdir() if path.is_file() and path.suffix == ".txt"
    }
    target_names = {
        path.name for path in target_path.iterdir() if path.is_file() and path.suffix == ".txt"
    }

    missing_names = sorted(reference_names - target_names)
    created_paths: list[Path] = []
    for name in missing_names:
        output_path = target_path / name
        output_path.touch(exist_ok=False)
        created_paths.append(output_path)

    return created_paths


if __name__ == "__main__":
    created = create_missing_track_txt()
    print(f"Created {len(created)} missing txt files.")
    for path in created:
        print(path)
