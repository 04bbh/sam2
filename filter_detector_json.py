#!/usr/bin/env python3
"""Filter detector JSON files by confidence and category frequency."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "output_json_detector/detector_stage2_json_chaserunmove_delchaseclass"
)
DEFAULT_OUTPUT_DIR = Path(
    "output_json_detector/detector_stage2_json_chaserunmove_delchaseclass_filtered"
)
CONFIDENCE_THRESHOLD = 0.9
MIN_CATEGORY_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter each detector JSON file. Empty detections are removed; "
            "detections with confidence < threshold are removed; categories "
            "with at least min-count remaining detections are reduced to the "
            "highest-confidence detection, breaking ties by the smallest frame_id."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing source JSON files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for filtered JSON files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Drop detections with confidence below this value. Default: {CONFIDENCE_THRESHOLD}",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=MIN_CATEGORY_COUNT,
        help=(
            "Only categories with at least this many detections after confidence "
            f"filtering are kept. Default: {MIN_CATEGORY_COUNT}"
        ),
    )
    return parser.parse_args()


def confidence_value(detection: dict[str, Any]) -> float:
    return float(detection.get("confidence", 0.0))


def frame_id_value(block: dict[str, Any]) -> int:
    return int(block.get("frame_id", 0))


def filter_low_confidence_blocks(
    blocks: list[dict[str, Any]], threshold: float
) -> list[dict[str, Any]]:
    filtered_blocks: list[dict[str, Any]] = []

    for block in blocks:
        detections = block.get("detections", [])
        if not detections:
            continue

        kept_detections = [
            detection
            for detection in detections
            if confidence_value(detection) >= threshold
        ]
        if not kept_detections:
            continue

        filtered_block = copy.deepcopy(block)
        filtered_block["detections"] = copy.deepcopy(kept_detections)
        filtered_blocks.append(filtered_block)

    return filtered_blocks


def pick_category_representatives(
    blocks: list[dict[str, Any]], min_count: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], int]]] = defaultdict(list)

    for block_index, block in enumerate(blocks):
        for detection in block.get("detections", []):
            category = detection.get("category")
            if category is None:
                continue
            grouped[str(category)].append((block, detection, block_index))

    selected_by_block: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for items in grouped.values():
        if len(items) < min_count:
            continue

        winner_block, winner_detection, winner_block_index = max(
            items,
            key=lambda item: (
                confidence_value(item[1]),
                -frame_id_value(item[0]),
                -item[2],
            ),
        )

        if winner_block_index not in selected_by_block:
            selected_block = copy.deepcopy(winner_block)
            selected_block["detections"] = []
            selected_by_block[winner_block_index] = (selected_block, [])

        selected_by_block[winner_block_index][1].append(copy.deepcopy(winner_detection))

    result: list[dict[str, Any]] = []
    for block_index in sorted(selected_by_block):
        selected_block, detections = selected_by_block[block_index]
        selected_block["detections"] = detections
        result.append(selected_block)

    return result


def filter_json_data(
    data: list[dict[str, Any]], threshold: float, min_count: int
) -> list[dict[str, Any]]:
    confidence_filtered = filter_low_confidence_blocks(data, threshold)
    return pick_category_representatives(confidence_filtered, min_count)


def process_file(input_path: Path, output_path: Path, threshold: float, min_count: int) -> int:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a top-level JSON list")

    filtered_data = filter_json_data(data, threshold, min_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return len(filtered_data)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    total_kept_blocks = 0
    for input_path in json_paths:
        output_path = output_dir / input_path.name
        total_kept_blocks += process_file(
            input_path=input_path,
            output_path=output_path,
            threshold=args.threshold,
            min_count=args.min_count,
        )

    print(
        f"Processed {len(json_paths)} files. "
        f"Kept {total_kept_blocks} blocks in {output_dir}."
    )


if __name__ == "__main__":
    main()
