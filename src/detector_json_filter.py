#!/usr/bin/env python3
"""Utilities for filtering detector JSON files."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("output_json_detector/detector_stage2_json_chaserunmove_1")
DEFAULT_OUTPUT_DIR = Path("output_json_detector/detector_stage2_json_chaserunmove_1_filtered")
CONFIDENCE_THRESHOLD = 0.9
MIN_CATEGORY_COUNT = 3
RANGE_CONFIDENCE_THRESHOLD = 0.85

FULL_VIDEO_RANGE_CATEGORIES = {
    "骑自行车",
    "骑摩托车",
    "机动车",
    "小推车",
    "推车",
    "垃圾推车",
}
ACTION_RANGE_CATEGORIES = {
    "打斗",
    "跳跃",
    "抢夺",
    "翻越栏杆",
    "摔倒",
    "奔跑",
    "快速奔跑",
    "追逐",
    "挥舞物品",
    "滑滑板",
    "滑滑板的人",
    "向上抛物品",
    "捡起掉落的物品"
}
def confidence_value(detection: dict[str, Any]) -> float:
    try:
        return float(detection.get("confidence", 0.0))
    except Exception:
        return 0.0


def frame_id_value(block: dict[str, Any]) -> int:
    try:
        return int(block.get("frame_id", 0))
    except Exception:
        return 0


def normalize_category_name(category: Any) -> str:
    if category is None:
        return ""
    category_name = str(category).strip()
    if not category_name:
        return ""
    return category_name


def filter_low_confidence_blocks(
    blocks: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    filtered_blocks: list[dict[str, Any]] = []

    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list) or not detections:
            continue

        kept_detections = [
            detection
            for detection in detections
            if isinstance(detection, dict) and confidence_value(detection) >= threshold
        ]
        if not kept_detections:
            continue

        filtered_block = copy.deepcopy(block)
        filtered_block["detections"] = copy.deepcopy(kept_detections)
        filtered_blocks.append(filtered_block)

    return filtered_blocks


def pick_category_representatives(
    blocks: list[dict[str, Any]],
    min_count: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], int]]] = defaultdict(list)

    for block_index, block in enumerate(blocks):
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            category = detection.get("category")
            if category is None:
                continue
            category_name = str(category).strip()
            if not category_name:
                continue
            grouped[category_name].append((block, detection, block_index))

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


def pick_category_multi_best(
    blocks: list[dict[str, Any]],
    min_count: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any], int]]] = defaultdict(list)

    for block_index, block in enumerate(blocks):
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            category = detection.get("category")
            if category is None:
                continue
            category_name = str(category).strip()
            if not category_name:
                continue
            grouped[category_name].append((block, detection, block_index))

    selected_by_block: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for items in grouped.values():
        if len(items) < min_count:
            continue

        best_confidence = max(confidence_value(detection) for _, detection, _ in items)
        winners = [
            (block, detection, block_index)
            for block, detection, block_index in items
            if math.isclose(
                confidence_value(detection),
                best_confidence,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ]

        for winner_block, winner_detection, winner_block_index in winners:
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
    data: list[dict[str, Any]],
    threshold: float,
    min_count: int,
) -> list[dict[str, Any]]:
    confidence_filtered = filter_low_confidence_blocks(data, threshold)
    return pick_category_representatives(confidence_filtered, min_count)


def filter_json_data_multi_best(
    data: list[dict[str, Any]],
    threshold: float,
    min_count: int,
    video_last_frame_id: int,
    range_threshold: float = RANGE_CONFIDENCE_THRESHOLD,
    enable_temporal_localization: bool = True,
    enable_box_filter: bool = True,
    random_keyframe_seed: int = 0,
) -> list[dict[str, Any]]:
    if enable_box_filter:
        confidence_filtered = filter_low_confidence_blocks(data, threshold)
        filtered_multi_best = pick_category_multi_best(confidence_filtered, min_count)
    else:
        filtered_multi_best = pick_global_best_single(data, random_keyframe_seed)

    if enable_temporal_localization:
        category_ranges = compute_category_frame_ranges(
            original_blocks=data,
            selected_blocks=filtered_multi_best,
            video_last_frame_id=video_last_frame_id,
            range_threshold=range_threshold,
        )
        attach_category_ranges(filtered_multi_best, category_ranges)
    else:
        attach_full_video_ranges(
            blocks=filtered_multi_best,
            video_last_frame_id=video_last_frame_id,
        )
    return filtered_multi_best


def pick_global_best_single(
    blocks: list[dict[str, Any]],
    random_keyframe_seed: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    best_confidence: float | None = None

    for block_index, block in enumerate(blocks):
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            category_name = normalize_category_name(detection.get("category"))
            if not category_name:
                continue

            current_confidence = confidence_value(detection)
            if best_confidence is None or current_confidence > best_confidence + 1e-9:
                best_confidence = current_confidence
                candidates = [(block, detection, block_index)]
            elif best_confidence is not None and math.isclose(
                current_confidence,
                best_confidence,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                candidates.append((block, detection, block_index))

    if not candidates:
        return []

    rng = random.Random(int(random_keyframe_seed))
    winner_block, winner_detection, _ = rng.choice(candidates)
    selected_block = copy.deepcopy(winner_block)
    selected_block["detections"] = [copy.deepcopy(winner_detection)]
    return [selected_block]


def collect_selected_categories(
    blocks: list[dict[str, Any]],
) -> set[str]:
    selected_categories: set[str] = set()
    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            category_name = normalize_category_name(detection.get("category"))
            if category_name:
                selected_categories.add(category_name)
    return selected_categories


def compute_category_frame_ranges(
    original_blocks: list[dict[str, Any]],
    selected_blocks: list[dict[str, Any]],
    video_last_frame_id: int,
    range_threshold: float,
) -> dict[str, tuple[int, int]]:
    if not original_blocks or video_last_frame_id < 0:
        return {}

    selected_categories = collect_selected_categories(selected_blocks)
    if not selected_categories:
        return {}

    category_hit_indices: dict[str, list[int]] = defaultdict(list)
    for block_index, block in enumerate(original_blocks):
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            category_name = normalize_category_name(detection.get("category"))
            if category_name not in selected_categories:
                continue
            if confidence_value(detection) < range_threshold:
                continue
            category_hit_indices[category_name].append(block_index)

    category_ranges: dict[str, tuple[int, int]] = {}
    for category_name in selected_categories:
        if category_name in FULL_VIDEO_RANGE_CATEGORIES:
            category_ranges[category_name] = (0, video_last_frame_id)
            continue

        if category_name not in ACTION_RANGE_CATEGORIES:
            continue

        hit_indices = category_hit_indices.get(category_name, [])
        if not hit_indices:
            category_ranges[category_name] = (0, video_last_frame_id)
            continue

        first_hit_index = hit_indices[0]
        last_hit_index = hit_indices[-1]

        first_hit_frame_id = frame_id_value(original_blocks[first_hit_index])
        if first_hit_index > 0:
            start_frame_id = frame_id_value(original_blocks[first_hit_index - 1])
        else:
            start_frame_id = max(first_hit_frame_id - 30, 0)

        end_index = min(last_hit_index + 5, len(original_blocks) - 1)
        end_frame_id = frame_id_value(original_blocks[end_index])

        start_frame_id = max(0, start_frame_id)
        end_frame_id = max(start_frame_id, end_frame_id)
        category_ranges[category_name] = (start_frame_id, end_frame_id)

    return category_ranges


def attach_category_ranges(
    blocks: list[dict[str, Any]],
    category_ranges: dict[str, tuple[int, int]],
) -> None:
    if not category_ranges:
        return

    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            category_name = normalize_category_name(detection.get("category"))
            if not category_name or category_name not in category_ranges:
                continue
            start_frame_id, end_frame_id = category_ranges[category_name]
            detection["start_frame_id"] = int(start_frame_id)
            detection["end_frame_id"] = int(end_frame_id)


def attach_full_video_ranges(
    blocks: list[dict[str, Any]],
    video_last_frame_id: int,
) -> None:
    end_frame_id = max(0, int(video_last_frame_id))
    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            detection["start_frame_id"] = 0
            detection["end_frame_id"] = end_frame_id


def process_detector_json(
    input_path: Path,
    output_path: Path | None,
    threshold: float,
    min_count: int,
) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a top-level JSON list")

    filtered_data = filter_json_data(data, threshold, min_count)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(filtered_data, file, ensure_ascii=False, indent=2)
            file.write("\n")

    return filtered_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter detector JSON files. Empty detections are removed, low "
            "confidence detections are dropped, and each frequent category is "
            "reduced to its highest-confidence representative."
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


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    json_paths = sorted(args.input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {args.input_dir}")

    total_kept_blocks = 0
    for input_path in json_paths:
        output_path = args.output_dir / input_path.name
        filtered_data = process_detector_json(
            input_path=input_path,
            output_path=output_path,
            threshold=args.threshold,
            min_count=args.min_count,
        )
        total_kept_blocks += len(filtered_data)

    print(
        f"Processed {len(json_paths)} files. "
        f"Kept {total_kept_blocks} blocks in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
