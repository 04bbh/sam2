#!/usr/bin/env python3
"""Run filtering, segmentation, and tracking from saved detector JSON files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import yaml

from src.dataset_target_config import DatasetTargetConfig, get_dataset_target_config
from src.detector_json_filter import (
    CONFIDENCE_THRESHOLD,
    MIN_CATEGORY_COUNT,
    RANGE_CONFIDENCE_THRESHOLD,
    filter_json_data_multi_best,
)
from src.json_segmenter_and_tracker import JsonSegmenterAndTracker, list_frames
from utils.mask_to_box_track import save_video_track_txt
from utils.visualization import save_tracking_results


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read saved detector JSON files, filter detections, and use the "
            "remaining boxes as SAM2 segmentation/tracking seeds."
        )
    )
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument(
        "--input-json-dir",
        type=Path,
        default="./output_detector_json/output_json_detector_xd/detector_stage2_json_2",
        help="Directory containing detector JSON files. Defaults to data.detector_stage2_json_root.",
    )
    parser.add_argument(
        "--filtered-json-root",
        type=Path,
        default="filtered_1",
        help="Directory for filtered JSON files. Defaults to ${input-json-dir}_filtered2.",
    )
    parser.add_argument(
        "--save-filtered-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save filtered JSON files. Enabled by default.",
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
            "Only categories with at least this many detections after confidence"
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
            "Enabled by default; use --no-temporal-localization for ablation."
        ),
    )
    parser.add_argument(
        "--sam-iou-keyframe-selection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Select one keyframe per category by SAM IoU/confidence rules. "
            "Enabled by default; use --no-sam-iou-keyframe-selection for random ablation."
        ),
    )
    parser.add_argument(
        "--random-keyframe-seed",
        type=int,
        default=0,
        help="Random seed used when --no-sam-iou-keyframe-selection is set. Default: 0.",
    )
    return parser.parse_args()


def default_filtered_root(input_json_dir: Path) -> Path:
    return input_json_dir.with_name(input_json_dir.name + "_filtered")


def load_detector_json(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a top-level JSON list")
    return data


def count_detections(blocks: list[dict]) -> int:
    total = 0
    for block in blocks:
        detections = block.get("detections", [])
        if isinstance(detections, list):
            total += sum(1 for det in detections if isinstance(det, dict))
    return total


def count_categories(blocks: list[dict]) -> int:
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


def mean_confidence(blocks: list[dict]) -> float:
    total = 0.0
    count = 0

    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, dict):
                continue
            try:
                total += float(det.get("confidence", 0.0))
            except Exception:
                total += 0.0
            count += 1

    return total / count if count > 0 else 0.0


def save_filtered_json(blocks: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(blocks, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run_single_json(
    json_path: Path,
    filtered_json_root: Path,
    cfg: dict,
    segmenter_and_tracker: JsonSegmenterAndTracker,
    threshold: float,
    min_count: int,
    box_filter: bool,
    range_threshold: float,
    temporal_localization: bool,
    sam_iou_keyframe_selection: bool,
    random_keyframe_seed: int,
    save_filtered: bool,
    dataset_target_config: DatasetTargetConfig,
) -> None:
    video_name = json_path.stem
    videos_root = cfg["data"]["videos_root"]
    video_dir = os.path.join(videos_root, video_name)
    out_dir = os.path.join(cfg["data"]["output_root"], video_name)
    track_txt_path = os.path.join(cfg["data"]["track_txt_root"], video_name + ".txt")

    if not os.path.isdir(video_dir):
        print(f"[Skip] 视频目录不存在: {video_dir}")
        return
    if os.path.isdir(out_dir) and os.path.isfile(track_txt_path):
        print(f"[Skip] 输出目录和轨迹文件已存在: {out_dir} {track_txt_path}")
        return

    frame_paths = list_frames(video_dir)
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

    if not filtered_data:
        print(
            f"[Skip] 筛选后无检测: {video_name} "
            f"raw_blocks={len(data)} raw_detections={count_detections(data)}"
        )
        return

    merge_iou_thresh = float(cfg.get("selector", {}).get("merge_iou_thresh", 0.8))
    ann_obj_id = int(cfg["data"].get("ann_obj_id", 1))
    video_segments = segmenter_and_tracker.segment_and_track_best_iou(
        video_dir=video_dir,
        blocks=filtered_data,
        start_obj_id=ann_obj_id,
        merge_iou_thresh=merge_iou_thresh,
        use_sam_iou_keyframe_selection=sam_iou_keyframe_selection,
        random_keyframe_seed=random_keyframe_seed,
    )

    if not video_segments:
        print(f"[Skip] 未得到可用跟踪结果: {video_name}")
        return

    track_score = mean_confidence(filtered_data)
    save_tracking_results(video_dir=video_dir, video_segments=video_segments, out_dir=out_dir)
    save_video_track_txt(video_segments=video_segments, txt_path=track_txt_path, score=track_score)

    print(
        f"[Done] video={video_name} raw_blocks={len(data)} "
        f"filtered_blocks={len(filtered_data)} categories={count_categories(filtered_data)} "
        f"seeds={count_detections(filtered_data)} tracked_frames={len(video_segments)} "
        f"score={track_score:.3f}"
    )
    print(f"[Done] 保存结果: {out_dir}")
    print(f"[Done] 轨迹文件: {track_txt_path}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    dataset_target_config = get_dataset_target_config(cfg["data"]["dataset_name"])

    input_json_dir = args.input_json_dir
    if input_json_dir is None:
        input_json_dir = Path(cfg["data"]["detector_stage2_json_root"])
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
        f"action_categories={len(dataset_target_config.action_range_categories)}"
    )

    segmenter_and_tracker = JsonSegmenterAndTracker(
        model_cfg=cfg["sam2"]["model_cfg"],
        checkpoint=cfg["sam2"]["checkpoint"],
        multimask_output=cfg["sam2"]["multimask_output"],
        device=device,
        offload_video_to_cpu=bool(cfg["sam2"].get("offload_video_to_cpu", True)),
        offload_state_to_cpu=bool(cfg["sam2"].get("offload_state_to_cpu", True)),
        async_loading_frames=bool(cfg["sam2"].get("async_loading_frames", False)),
    )

    for json_path in json_paths:
        try:
            run_single_json(
                json_path=json_path,
                filtered_json_root=filtered_json_root,
                cfg=cfg,
                segmenter_and_tracker=segmenter_and_tracker,
                threshold=args.threshold,
                min_count=args.min_count,
                box_filter=args.box_filter,
                range_threshold=args.range_threshold,
                temporal_localization=args.temporal_localization,
                sam_iou_keyframe_selection=args.sam_iou_keyframe_selection,
                random_keyframe_seed=args.random_keyframe_seed,
                save_filtered=args.save_filtered_json,
                dataset_target_config=dataset_target_config,
            )
        except Exception as exc:
            print(f"[Error] {json_path.name}: {exc}")


if __name__ == "__main__":
    main()
