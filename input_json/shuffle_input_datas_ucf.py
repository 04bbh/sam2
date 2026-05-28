import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shuffle non-Normal videos in input_datas_ucf.json and place Normal "
            "videos at the end."
        )
    )
    parser.add_argument(
        "--input",
        default="input_json/input_datas_ucf.json",
        type=Path,
        help="Input JSON path.",
    )
    parser.add_argument(
        "--output",
        default="input_json/input_datas_ucf_shuffled.json",
        type=Path,
        help="Output JSON path. Ignored when --in-place is set.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional random seed for reproducible shuffling.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSON instead of writing to --output.",
    )
    return parser.parse_args()


def load_json_list(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level JSON list")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"{path} must contain only JSON objects in the top-level list")

    return data


def save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def main() -> int:
    args = parse_args()
    input_path = args.input
    output_path = input_path if args.in_place else args.output

    data = load_json_list(input_path)
    non_normal_items = [
        item for item in data if "Normal" not in str(item.get("video_path", ""))
    ]
    normal_items = [
        item for item in data if "Normal" in str(item.get("video_path", ""))
    ]

    random.Random(args.seed).shuffle(non_normal_items)
    shuffled_data = non_normal_items + normal_items
    save_json(shuffled_data, output_path)

    print(
        "[Done] shuffled "
        f"{len(non_normal_items)} non-Normal items and appended "
        f"{len(normal_items)} Normal items: {input_path} -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
