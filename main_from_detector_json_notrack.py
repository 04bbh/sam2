#!/usr/bin/env python3
"""Export merged temporal intervals from saved detector JSON files without tracking."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent


def load_repo_module(module_name: str, relative_path: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


dataset_target_config_module = load_repo_module(
    "dataset_target_config_notrack",
    "src/dataset_target_config.py",
)
detector_json_filter_module = load_repo_module(
    "detector_json_filter_notrack",
    "src/detector_json_filter.py",
)
get_dataset_target_config = dataset_target_config_module.get_dataset_target_config
CONFIDENCE_THRESHOLD = detector_json_filter_module.CONFIDENCE_THRESHOLD
MIN_CATEGORY_COUNT = detector_json_filter_module.MIN_CATEGORY_COUNT
RANGE_CONFIDENCE_THRESHOLD = detector_json_filter_module.RANGE_CONFIDENCE_THRESHOLD
filter_json_data_multi_best = detector_json_filter_module.filter_json_data_multi_best


DEFAULT_INPUT_JSON_DIR = Path(
    "./output_detector_json/output_json_detector_xd/detector_stage2_json_1"
)
DEFAULT_OUTPUT_TXT_ROOT = Path(
    "/data/users/wjq/codes/sam2-main/output_tracks/output_tracks_xd/improved_1_notrack"
)
DEFAULT_TRACK_TXT_ROOT = Path(
    "/data/users/wjq/codes/sam2-main/output_tracks/output_tracks_xd/improved_1"
)
DEFAULT_VIDEOS_ROOT = Path("/data/users/wjq/datasets/XD/frames")
DEFAULT_DATASET_NAME = "xd"
DEFAULT_FRAME_INDEX_SCALE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read saved detector JSON files, filter detections, merge their "
            "temporal ranges, and export one interval txt per video."
        )
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        type=str,
        help=f"Dataset name for target category range rules. Default: {DEFAULT_DATASET_NAME}",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help=f"Directory containing video frame folders. Default: {DEFAULT_VIDEOS_ROOT}",
    )
    parser.add_argument(
        "--input-json-dir",
        type=Path,
        default=DEFAULT_INPUT_JSON_DIR,
        help="Directory containing detector JSON files. Defaults to the XD detector JSON root.",
    )
    parser.add_argument(
        "--output-txt-root",
        type=Path,
        default=DEFAULT_OUTPUT_TXT_ROOT,
        help=f"Directory for merged interval txt files. Default: {DEFAULT_OUTPUT_TXT_ROOT}",
    )
    parser.add_argument(
        "--track-txt-root",
        type=Path,
        default=DEFAULT_TRACK_TXT_ROOT,
        help=(
            "Directory containing per-frame track txt files. When --interval-source=track, "
            f"the first column is converted to continuous intervals. Default: {DEFAULT_TRACK_TXT_ROOT}"
        ),
    )
    parser.add_argument(
        "--interval-source",
        choices=("track", "detector"),
        default="track",
        help=(
            "Source used to build temporal intervals. 'track' reads existing per-frame "
            "track txt files; 'detector' merges detector JSON temporal ranges. Default: track."
        ),
    )
    parser.add_argument(
        "--frame-index-scale",
        type=int,
        default=DEFAULT_FRAME_INDEX_SCALE,
        help=(
            "Divide interval endpoints by this value with integer division before writing. "
            f"Use 1 to keep raw frame ids. Default: {DEFAULT_FRAME_INDEX_SCALE}."
        ),
    )
    parser.add_argument(
        "--filtered-json-root",
        type=Path,
        default="filtered_1",
        help="Directory for optional filtered JSON files.",
    )
    parser.add_argument(
        "--save-filtered-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save filtered JSON files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help=f"Drop detections with confidence below this value. Default: {CONFIDENCE_THRESHOLD}",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=3,
        help=(
            "Only categories with at least this many detections after confidence "
            f"filtering are kept. Default: {MIN_CATEGORY_COUNT}"
        ),
    )
    parser.add_argument(
        "--box-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable confidence-threshold box filtering before multi-best selection. "
            "Enabled by default; use --no-box-filter for ablation."
        ),
    )
    parser.add_argument(
        "--range-threshold",
        type=float,
        default=0.85,
        help=(
            "Confidence threshold used only for category start/end frame range "
            f"calculation. Default: {RANGE_CONFIDENCE_THRESHOLD}"
        ),
    )
    parser.add_argument(
        "--temporal-localization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable category temporal localization for start/end frame ids. "
            "Enabled by default; use --no-temporal-localization for full-video ranges."
        ),
    )
    parser.add_argument(
        "--random-keyframe-seed",
        type=int,
        default=0,
        help="Random seed used when --no-box-filter is set. Default: 0.",
    )
    return parser.parse_args()


def default_filtered_root(input_json_dir: Path) -> Path:
    return input_json_dir.with_name(input_json_dir.name + "_filtered")


def list_frames(video_dir: str) -> list[str]:
    names = [
        name
        for name in os.listdir(video_dir)
        if os.path.splitext(name)[-1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    names.sort()
    return [os.path.join(video_dir, name) for name in names]


def load_detector_json(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a top-level JSON list")
    return data


def count_detections(blocks: list[dict[str, Any]]) -> int:
    total = 0
    for block in blocks:
        detections = block.get("detections", [])
        if isinstance(detections, list):
            total += sum(1 for det in detections if isinstance(det, dict))
    return total


def count_categories(blocks: list[dict[str, Any]]) -> int:
    categories = set()
    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            category = det.get("category")
            if isinstance(category, str) and category.strip():
                categories.add(category.strip())
    return len(categories)


def save_filtered_json(blocks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(blocks, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_int_field(detection: dict[str, Any], field_name: str) -> int | None:
    value = detection.get(field_name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def collect_detection_intervals(
    blocks: list[dict[str, Any]],
    video_last_frame_id: int,
) -> tuple[list[tuple[int, int]], int, int]:
    intervals: list[tuple[int, int]] = []
    missing_range_count = 0
    invalid_range_count = 0
    last_frame = max(0, int(video_last_frame_id))

    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue

            start_frame_id = parse_int_field(detection, "start_frame_id")
            end_frame_id = parse_int_field(detection, "end_frame_id")
            if start_frame_id is None or end_frame_id is None:
                missing_range_count += 1
                continue

            start = min(max(start_frame_id, 0), last_frame)
            end = min(max(end_frame_id, 0), last_frame)
            if end < start:
                invalid_range_count += 1
                continue
            intervals.append((start, end))

    return intervals, missing_range_count, invalid_range_count


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int]] = []
    current_start, current_end = sorted_intervals[0]

    for start, end in sorted_intervals[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged


def read_track_frame_indices(track_txt_path: Path) -> list[int]:
    frame_indices: set[int] = set()
    with track_txt_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            first_column = line.split(",", maxsplit=1)[0].strip()
            try:
                frame_indices.add(int(first_column))
            except ValueError as error:
                raise ValueError(
                    f"{track_txt_path} line {line_number} has invalid frame index: "
                    f"{first_column!r}"
                ) from error
    return sorted(frame_indices)


def frame_indices_to_intervals(frame_indices: list[int]) -> list[tuple[int, int]]:
    if not frame_indices:
        return []

    intervals: list[tuple[int, int]] = []
    start = previous = frame_indices[0]
    for frame_index in frame_indices[1:]:
        if frame_index == previous + 1:
            previous = frame_index
            continue
        intervals.append((start, previous))
        start = previous = frame_index
    intervals.append((start, previous))
    return intervals


def scale_intervals(
    intervals: list[tuple[int, int]],
    frame_index_scale: int,
) -> list[tuple[int, int]]:
    if frame_index_scale <= 0:
        raise ValueError("--frame-index-scale must be > 0")
    if frame_index_scale == 1:
        return intervals
    return [
        (start // frame_index_scale, end // frame_index_scale)
        for start, end in intervals
    ]


def save_interval_txt(
    intervals: list[tuple[int, int]],
    txt_path: Path,
    frame_index_scale: int,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    output_intervals = scale_intervals(intervals, frame_index_scale)
    lines = [f"{start},{end}" for start, end in output_intervals]
    with txt_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))
        if lines:
            file.write("\n")


def run_single_json(
    json_path: Path,
    output_txt_root: Path,
    track_txt_root: Path,
    filtered_json_root: Path,
    videos_root: Path,
    interval_source: str,
    frame_index_scale: int,
    threshold: float,
    min_count: int,
    box_filter: bool,
    range_threshold: float,
    temporal_localization: bool,
    random_keyframe_seed: int,
    save_filtered: bool,
    dataset_target_config: Any,
) -> None:
    video_name = json_path.stem
    video_dir = videos_root / video_name
    interval_txt_path = output_txt_root / f"{video_name}.txt"
    track_txt_path = track_txt_root / f"{video_name}.txt"

    if interval_source == "track":
        if not track_txt_path.is_file():
            print(f"[Skip] track txt 不存在: {track_txt_path}")
            return

        frame_indices = read_track_frame_indices(track_txt_path)
        intervals = frame_indices_to_intervals(frame_indices)
        save_interval_txt(intervals, interval_txt_path, frame_index_scale)
        print(
            f"[Done] video={video_name} track_frames={len(frame_indices)} "
            f"intervals={len(intervals)} scale={frame_index_scale} "
            f"txt={interval_txt_path}"
        )
        return

    if not video_dir.is_dir():
        print(f"[Skip] 视频目录不存在: {video_dir}")
        return

    frame_paths = list_frames(str(video_dir))
    if not frame_paths:
        print(f"[Skip] 视频目录无可用帧: {video_dir}")
        return
    video_last_frame_id = len(frame_paths) - 1

    data = load_detector_json(json_path)
    filtered_data = filter_json_data_multi_best(
        data,
        threshold=threshold,
        min_count=min_count,
        video_last_frame_id=video_last_frame_id,
        range_threshold=range_threshold,
        enable_temporal_localization=temporal_localization,
        enable_box_filter=box_filter,
        random_keyframe_seed=random_keyframe_seed,
        full_video_range_categories=dataset_target_config.full_video_range_categories,
        action_range_categories=dataset_target_config.action_range_categories,
    )

    if save_filtered:
        filtered_path = filtered_json_root / json_path.name
        save_filtered_json(filtered_data, filtered_path)
        print(f"[Done] 筛选 JSON: {filtered_path}")

    intervals, missing_range_count, invalid_range_count = collect_detection_intervals(
        blocks=filtered_data,
        video_last_frame_id=video_last_frame_id,
    )
    merged_intervals = merge_intervals(intervals)
    save_interval_txt(merged_intervals, interval_txt_path, frame_index_scale)

    if not filtered_data:
        print(
            f"[Skip] 筛选后无检测: {video_name} "
            f"raw_blocks={len(data)} raw_detections={count_detections(data)} "
            f"txt={interval_txt_path}"
        )
        return

    if missing_range_count or invalid_range_count:
        print(
            f"[Warn] video={video_name} missing_ranges={missing_range_count} "
            f"invalid_ranges={invalid_range_count}"
        )

    print(
        f"[Done] video={video_name} raw_blocks={len(data)} "
        f"filtered_blocks={len(filtered_data)} categories={count_categories(filtered_data)} "
        f"seeds={count_detections(filtered_data)} intervals={len(merged_intervals)} "
        f"scale={frame_index_scale} txt={interval_txt_path}"
    )


def main() -> None:
    args = parse_args()
    dataset_target_config = get_dataset_target_config(args.dataset_name)

    input_json_dir = args.input_json_dir
    filtered_json_root = args.filtered_json_root or default_filtered_root(input_json_dir)

    if not input_json_dir.is_dir():
        raise FileNotFoundError(f"Input JSON directory does not exist: {input_json_dir}")

    json_paths = sorted(input_json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {input_json_dir}")

    print(
        f"[Info] dataset={dataset_target_config.name} "
        f"targets={len(dataset_target_config.target_desc.split('，'))} "
        f"full_video_categories={len(dataset_target_config.full_video_range_categories)} "
        f"action_categories={len(dataset_target_config.action_range_categories)} "
        f"interval_source={args.interval_source} "
        f"frame_index_scale={args.frame_index_scale} "
        f"output_txt_root={args.output_txt_root}"
    )

    for json_path in json_paths:
        try:
            run_single_json(
                json_path=json_path,
                output_txt_root=args.output_txt_root,
                track_txt_root=args.track_txt_root,
                filtered_json_root=filtered_json_root,
                videos_root=args.videos_root,
                interval_source=args.interval_source,
                frame_index_scale=args.frame_index_scale,
                threshold=args.threshold,
                min_count=args.min_count,
                box_filter=args.box_filter,
                range_threshold=args.range_threshold,
                temporal_localization=args.temporal_localization,
                random_keyframe_seed=args.random_keyframe_seed,
                save_filtered=args.save_filtered_json,
                dataset_target_config=dataset_target_config,
            )
        except Exception as exc:
            print(f"[Error] {json_path.name}: {exc}")


if __name__ == "__main__":
    main()
