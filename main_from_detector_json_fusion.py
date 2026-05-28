#!/usr/bin/env python3
"""Run detector-JSON fusion scoring, temporal proposal, and SAM2 tracking."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class JsonDetectionCandidate:
    block_index: int
    detection_index: int
    frame_id: int
    frame_path: str
    category: str
    box_xyxy: np.ndarray
    det_conf: float
    frame_score: float
    category_presence_score: float
    category_presence: bool
    sam_iou_score: float = 0.0
    object_score: float = 0.0
    selected_as_key: bool = False
    start_frame_id: int | None = None
    end_frame_id: int | None = None
    mask: np.ndarray | None = None


def load_cfg(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read saved detector JSON files, fuse category/object/frame scores, "
            "generate temporal proposals, and track selected SAM2 masks."
        )
    )
    parser.add_argument(
        "--config",
        default="config_from_detector_json_fusion.yaml",
        type=str,
        help="Fusion pipeline config path.",
    )
    return parser.parse_args()


def load_detector_json(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} must contain a top-level JSON list")
    return data


def load_video_scores(input_json_path: str | Path) -> dict[str, list[float]]:
    with open(input_json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{input_json_path} must contain a top-level JSON list")

    scores_by_video: dict[str, list[float]] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        video_path = item.get("video_path", "")
        if not isinstance(video_path, str) or not video_path:
            continue
        raw_scores = item.get("vid_bin_scores", [])
        if not isinstance(raw_scores, list):
            raw_scores = []
        scores: list[float] = []
        for score in raw_scores:
            try:
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        scores_by_video[video_path] = scores
    return scores_by_video


def list_frames(video_dir: str) -> list[str]:
    names = [
        name
        for name in os.listdir(video_dir)
        if os.path.splitext(name)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]

    def sort_key(name: str) -> tuple[int, int | str]:
        stem = os.path.splitext(name)[0]
        try:
            return (0, int(stem))
        except Exception:
            return (1, name)

    names.sort(key=sort_key)
    return [os.path.join(video_dir, name) for name in names]


def normalize_category(category: Any) -> str:
    if category is None:
        return ""
    return str(category).strip()


def confidence_value(detection: dict[str, Any]) -> float:
    try:
        return float(detection.get("confidence", 0.0))
    except Exception:
        return 0.0


def frame_id_value(block: dict[str, Any]) -> int:
    try:
        return int(block.get("frame_id", block.get("frame_idx", -1)))
    except Exception:
        return -1


def parse_box(box: Any) -> np.ndarray:
    if not isinstance(box, list) or len(box) != 4:
        return np.zeros(4, dtype=np.float32)
    try:
        box_arr = np.asarray(box, dtype=np.float32)
    except Exception:
        return np.zeros(4, dtype=np.float32)
    if not np.isfinite(box_arr).all():
        return np.zeros(4, dtype=np.float32)
    return box_arr


def is_valid_box(box_xyxy: np.ndarray) -> bool:
    if box_xyxy.shape != (4,):
        return False
    if np.allclose(box_xyxy, np.zeros(4, dtype=np.float32)):
        return False
    x1, y1, x2, y2 = [float(x) for x in box_xyxy]
    return x2 > x1 and y2 > y1


def frame_score_for_id(
    frame_id: int,
    scores: list[float],
    candidate_stride: int,
    frame_score_window_size: int = 0,
) -> float:
    if not scores or candidate_stride <= 0:
        return 0.0
    score_idx = int(math.floor(float(frame_id) / float(candidate_stride)))
    score_idx = max(0, min(len(scores) - 1, score_idx))
    window_size = max(0, int(frame_score_window_size))
    start_idx = max(0, score_idx - window_size)
    end_idx = min(len(scores) - 1, score_idx + window_size)
    window_scores = scores[start_idx : end_idx + 1]
    return float(sum(window_scores) / len(window_scores)) if window_scores else 0.0


def compute_category_presence(
    blocks: list[dict[str, Any]],
    top_k: int,
    threshold: float,
    padded_value: float,
) -> tuple[dict[str, float], dict[str, bool]]:
    grouped: dict[str, list[float]] = {}
    for block in blocks:
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            category = normalize_category(detection.get("category"))
            if not category:
                continue
            grouped.setdefault(category, []).append(confidence_value(detection))

    presence_scores: dict[str, float] = {}
    presence_flags: dict[str, bool] = {}
    for category, values in grouped.items():
        sorted_values = sorted(values, reverse=True)
        selected = sorted_values[:top_k]
        if len(selected) < top_k:
            selected.extend([float(padded_value)] * (top_k - len(selected)))
        score = float(sum(selected) / float(top_k)) if top_k > 0 else 0.0
        presence_scores[category] = score
        presence_flags[category] = score >= float(threshold)

    return presence_scores, presence_flags


def collect_candidates(
    blocks: list[dict[str, Any]],
    frame_paths: list[str],
    scores: list[float],
    candidate_stride: int,
    frame_score_window_size: int,
    presence_scores: dict[str, float],
    presence_flags: dict[str, bool],
) -> list[JsonDetectionCandidate]:
    candidates: list[JsonDetectionCandidate] = []
    for block_index, block in enumerate(blocks):
        frame_id = frame_id_value(block)
        if frame_id < 0 or frame_id >= len(frame_paths):
            continue
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            category = normalize_category(detection.get("category"))
            if not category or not presence_flags.get(category, False):
                continue
            det_conf = confidence_value(detection)
            box_xyxy = parse_box(detection.get("box"))
            if not is_valid_box(box_xyxy):
                continue
            candidates.append(
                JsonDetectionCandidate(
                    block_index=block_index,
                    detection_index=detection_index,
                    frame_id=frame_id,
                    frame_path=frame_paths[frame_id],
                    category=category,
                    box_xyxy=box_xyxy,
                    det_conf=det_conf,
                    frame_score=frame_score_for_id(
                        frame_id,
                        scores,
                        candidate_stride,
                        frame_score_window_size,
                    ),
                    category_presence_score=presence_scores.get(category, 0.0),
                    category_presence=True,
                )
            )
    return candidates


def segment_and_score_candidates(
    segmenter: Any,
    candidates: list[JsonDetectionCandidate],
    detection_result_cls: Any,
    sam_conf_threshold: float,
    det_conf_weight: float,
    frame_score_weight: float,
    sam_iou_weight: float,
) -> list[JsonDetectionCandidate]:
    sam_items = [
        candidate
        for candidate in candidates
        if candidate.det_conf >= float(sam_conf_threshold)
    ]
    detections = [
        detection_result_cls(
            frame_idx=candidate.frame_id,
            frame_path=candidate.frame_path,
            box_xyxy=candidate.box_xyxy,
            confidence=candidate.det_conf,
            category=candidate.category,
        )
        for candidate in sam_items
    ]
    segmentations = segmenter.segment_many(detections) if detections else []

    for candidate, segmentation in zip(sam_items, segmentations):
        mask = np.asarray(segmentation.mask)
        if np.allclose(segmentation.box_xyxy, np.zeros(4, dtype=np.float32)):
            continue
        if mask.sum() <= 0:
            continue
        candidate.sam_iou_score = float(segmentation.iou_score)
        candidate.mask = mask.astype(np.uint8)

    for candidate in candidates:
        candidate.object_score = (
            float(det_conf_weight) * candidate.det_conf
            + float(frame_score_weight) * candidate.frame_score
            + float(sam_iou_weight) * candidate.sam_iou_score
        )
    return candidates


def has_valid_mask(candidate: JsonDetectionCandidate) -> bool:
    if candidate.mask is None:
        return False
    mask = np.asarray(candidate.mask)
    return mask.ndim >= 2 and mask.sum() > 0


def select_key_candidates(
    candidates: list[JsonDetectionCandidate],
) -> dict[str, JsonDetectionCandidate]:
    selected: dict[str, JsonDetectionCandidate] = {}
    for candidate in candidates:
        if not has_valid_mask(candidate):
            continue
        current = selected.get(candidate.category)
        candidate_key = (
            -candidate.object_score,
            -candidate.det_conf,
            -candidate.sam_iou_score,
            candidate.frame_id,
            candidate.block_index,
            candidate.detection_index,
        )
        if current is None:
            selected[candidate.category] = candidate
            continue
        current_key = (
            -current.object_score,
            -current.det_conf,
            -current.sam_iou_score,
            current.frame_id,
            current.block_index,
            current.detection_index,
        )
        if candidate_key < current_key:
            selected[candidate.category] = candidate

    for candidate in selected.values():
        candidate.selected_as_key = True
    return selected


def category_sampled_frame_ids(blocks: list[dict[str, Any]], video_last_frame_id: int) -> list[int]:
    frame_ids = {
        frame_id_value(block)
        for block in blocks
        if 0 <= frame_id_value(block) <= video_last_frame_id
    }
    return sorted(frame_ids)


def category_response_by_frame(
    blocks: list[dict[str, Any]],
    category: str,
    sampled_frame_ids: list[int],
) -> dict[int, float]:
    response = {frame_id: 0.0 for frame_id in sampled_frame_ids}
    for block in blocks:
        frame_id = frame_id_value(block)
        if frame_id not in response:
            continue
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            if normalize_category(detection.get("category")) != category:
                continue
            response[frame_id] = max(response[frame_id], confidence_value(detection))
    return response


def generate_temporal_range(
    blocks: list[dict[str, Any]],
    category: str,
    keyframe_id: int,
    scores: list[float],
    candidate_stride: int,
    frame_score_window_size: int,
    video_last_frame_id: int,
    full_video_range_categories: set[str] | frozenset[str],
    action_range_categories: set[str] | frozenset[str],
    alpha: float,
    category_response_weight: float,
    frame_score_weight: float,
    start_margin_sampled_frames: int,
    end_margin_sampled_frames: int,
) -> tuple[int, int] | None:
    if category in full_video_range_categories:
        return 0, video_last_frame_id
    if category not in action_range_categories:
        return None

    sampled_frame_ids = category_sampled_frame_ids(blocks, video_last_frame_id)
    if not sampled_frame_ids:
        return None

    response = category_response_by_frame(blocks, category, sampled_frame_ids)
    proposal_scores: dict[int, float] = {}
    for frame_id in sampled_frame_ids:
        proposal_scores[frame_id] = (
            float(category_response_weight) * response.get(frame_id, 0.0)
            + float(frame_score_weight)
            * frame_score_for_id(
                frame_id,
                scores,
                candidate_stride,
                frame_score_window_size,
            )
        )
    max_score = max(proposal_scores.values()) if proposal_scores else 0.0
    if max_score <= 0.0:
        return None

    threshold = float(alpha) * max_score
    active_frame_ids = [
        frame_id
        for frame_id in sampled_frame_ids
        if proposal_scores.get(frame_id, 0.0) >= threshold
    ]
    if not active_frame_ids:
        return None

    start_active_frame_id = min(active_frame_ids)
    end_active_frame_id = max(active_frame_ids)
    frame_index_by_id = {
        frame_id: index
        for index, frame_id in enumerate(sampled_frame_ids)
    }
    start_margin = max(0, int(start_margin_sampled_frames))
    end_margin = max(0, int(end_margin_sampled_frames))
    start_index = max(0, frame_index_by_id[start_active_frame_id] - start_margin)
    end_index = min(
        len(sampled_frame_ids) - 1,
        frame_index_by_id[end_active_frame_id] + end_margin,
    )
    start_frame_id = sampled_frame_ids[start_index]
    end_frame_id = sampled_frame_ids[end_index]
    start_frame_id = min(start_frame_id, keyframe_id)
    end_frame_id = max(end_frame_id, keyframe_id)
    start_frame_id = max(0, min(video_last_frame_id, start_frame_id))
    end_frame_id = max(start_frame_id, min(video_last_frame_id, end_frame_id))
    return start_frame_id, end_frame_id


def save_scored_json(
    original_blocks: list[dict[str, Any]],
    output_path: Path,
    presence_scores: dict[str, float],
    presence_flags: dict[str, bool],
    scored_candidates: list[JsonDetectionCandidate],
) -> None:
    blocks = copy.deepcopy(original_blocks)
    candidate_by_index = {
        (candidate.block_index, candidate.detection_index): candidate
        for candidate in scored_candidates
    }

    for block_index, block in enumerate(blocks):
        detections = block.get("detections", [])
        if not isinstance(detections, list):
            continue
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            category = normalize_category(detection.get("category"))
            detection["category_presence_score"] = float(presence_scores.get(category, 0.0))
            detection["category_presence"] = int(bool(presence_flags.get(category, False)))
            detection["selected_as_key"] = False

            candidate = candidate_by_index.get((block_index, detection_index))
            if candidate is None:
                continue

            detection["frame_score"] = float(candidate.frame_score)
            detection["sam_iou_score"] = float(candidate.sam_iou_score)
            detection["object_score"] = float(candidate.object_score)
            detection["selected_as_key"] = bool(candidate.selected_as_key)
            if candidate.start_frame_id is not None:
                detection["start_frame_id"] = int(candidate.start_frame_id)
            if candidate.end_frame_id is not None:
                detection["end_frame_id"] = int(candidate.end_frame_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(blocks, file, ensure_ascii=False, indent=2)
        file.write("\n")


def track_key_candidates(
    segmenter: Any,
    video_dir: str,
    selected_candidates: list[JsonDetectionCandidate],
    start_obj_id: int,
    merge_iou_thresh: float,
    merge_video_segments_fn: Any,
    torch_module: Any,
) -> dict[int, dict[int, np.ndarray]]:
    if not selected_candidates:
        return {}

    video_segments: dict[int, dict[int, np.ndarray]] = {}
    state = segmenter.video_predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=segmenter.offload_video_to_cpu,
        offload_state_to_cpu=segmenter.offload_state_to_cpu,
        async_loading_frames=segmenter.async_loading_frames,
    )
    try:
        obj_id = int(start_obj_id)
        for candidate in selected_candidates:
            if candidate.mask is None:
                continue
            if candidate.start_frame_id is None or candidate.end_frame_id is None:
                continue
            tracked_segments = segmenter._track_from_masks_with_state(
                state=state,
                ann_frame_idx=int(candidate.frame_id),
                obj_ids=[obj_id],
                masks=[candidate.mask],
                start_frame_idx=int(candidate.start_frame_id),
                end_frame_idx=int(candidate.end_frame_id),
            )
            merge_video_segments_fn(
                target=video_segments,
                source=tracked_segments,
                iou_thresh=float(merge_iou_thresh),
            )
            obj_id += 1
    finally:
        segmenter.video_predictor.reset_state(state)
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    return video_segments


def mean_selected_object_score(selected_candidates: list[JsonDetectionCandidate]) -> float:
    if not selected_candidates:
        return 0.0
    return float(sum(candidate.object_score for candidate in selected_candidates) / len(selected_candidates))


def run_single_json(
    json_path: Path,
    scores_by_video: dict[str, list[float]],
    cfg: dict[str, Any],
    dataset_target_config: Any,
    segmenter: Any,
    detection_result_cls: Any,
    merge_video_segments_fn: Any,
    save_tracking_results_fn: Any,
    save_video_track_txt_fn: Any,
    torch_module: Any,
) -> None:
    video_name = json_path.stem
    scores = scores_by_video.get(video_name)
    if scores is None:
        print(f"[Skip] 未找到帧级异常分数: {video_name}")
        return

    video_dir = os.path.join(cfg["data"]["videos_root"], video_name)
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

    blocks = load_detector_json(json_path)
    presence_cfg = cfg.get("category_presence", {})
    presence_scores, presence_flags = compute_category_presence(
        blocks=blocks,
        top_k=int(presence_cfg.get("top_k", 3)),
        threshold=float(presence_cfg.get("threshold", 0.9)),
        padded_value=float(presence_cfg.get("padded_value", 0.0)),
    )
    existing_categories = [category for category, exists in presence_flags.items() if exists]
    if not existing_categories:
        print(f"[Skip] 无存在类别: {video_name}")
        return

    object_cfg = cfg.get("object_scoring", {})
    candidate_stride = int(cfg.get("data", {}).get("candidate_stride", 1))
    frame_score_window_size = int(cfg.get("data", {}).get("frame_score_window_size", 0))
    candidates = collect_candidates(
        blocks=blocks,
        frame_paths=frame_paths,
        scores=scores,
        candidate_stride=candidate_stride,
        frame_score_window_size=frame_score_window_size,
        presence_scores=presence_scores,
        presence_flags=presence_flags,
    )
    if not candidates:
        print(f"[Skip] 无有效合法候选: {video_name} categories={len(existing_categories)}")
        return

    sam_conf_threshold = float(
        object_cfg.get(
            "sam_conf_threshold",
            object_cfg.get("prefilter_conf_threshold", 0.85),
        )
    )
    sam_candidate_count = sum(1 for candidate in candidates if candidate.det_conf >= sam_conf_threshold)
    scored_candidates = segment_and_score_candidates(
        segmenter=segmenter,
        candidates=candidates,
        detection_result_cls=detection_result_cls,
        sam_conf_threshold=sam_conf_threshold,
        det_conf_weight=float(object_cfg.get("det_conf_weight", 0.8)),
        frame_score_weight=float(object_cfg.get("frame_score_weight", 0.1)),
        sam_iou_weight=float(object_cfg.get("sam_iou_weight", 0.1)),
    )
    if not scored_candidates:
        print(f"[Skip] 无可打分候选: {video_name}")
        return

    selected_by_category = select_key_candidates(scored_candidates)
    if not selected_by_category:
        print(
            f"[Skip] 无有效 key mask: {video_name} "
            f"candidates={len(candidates)} sam_candidates={sam_candidate_count}"
        )
        if bool(cfg.get("tracking", {}).get("save_scored_json", True)):
            scored_json_root = Path(cfg.get("tracking", {}).get("scored_json_root", "fusion_scored_json"))
            save_scored_json(
                original_blocks=blocks,
                output_path=scored_json_root / json_path.name,
                presence_scores=presence_scores,
                presence_flags=presence_flags,
                scored_candidates=scored_candidates,
            )
        return

    temporal_cfg = cfg.get("temporal_proposal", {})
    legacy_margin = int(temporal_cfg.get("margin_sampled_frames", 3))
    start_margin_sampled_frames = int(
        temporal_cfg.get("start_margin_sampled_frames", legacy_margin)
    )
    end_margin_sampled_frames = int(
        temporal_cfg.get("end_margin_sampled_frames", legacy_margin)
    )
    selected_candidates: list[JsonDetectionCandidate] = []
    skipped_categories = []
    for category, candidate in sorted(selected_by_category.items()):
        temporal_range = generate_temporal_range(
            blocks=blocks,
            category=category,
            keyframe_id=candidate.frame_id,
            scores=scores,
            candidate_stride=candidate_stride,
            frame_score_window_size=frame_score_window_size,
            video_last_frame_id=video_last_frame_id,
            full_video_range_categories=dataset_target_config.full_video_range_categories,
            action_range_categories=dataset_target_config.action_range_categories,
            alpha=float(temporal_cfg.get("alpha", 0.5)),
            category_response_weight=float(temporal_cfg.get("category_response_weight", 0.9)),
            frame_score_weight=float(temporal_cfg.get("frame_score_weight", 0.1)),
            start_margin_sampled_frames=start_margin_sampled_frames,
            end_margin_sampled_frames=end_margin_sampled_frames,
        )
        if temporal_range is None:
            candidate.selected_as_key = False
            skipped_categories.append(category)
            continue
        candidate.start_frame_id, candidate.end_frame_id = temporal_range
        selected_candidates.append(candidate)

    tracking_cfg = cfg.get("tracking", {})
    if bool(tracking_cfg.get("save_scored_json", True)):
        scored_json_root = Path(tracking_cfg.get("scored_json_root", "fusion_scored_json"))
        save_scored_json(
            original_blocks=blocks,
            output_path=scored_json_root / json_path.name,
            presence_scores=presence_scores,
            presence_flags=presence_flags,
            scored_candidates=scored_candidates,
        )

    if not selected_candidates:
        print(f"[Skip] 无可追踪类别: {video_name} skipped={skipped_categories}")
        return

    video_segments = track_key_candidates(
        segmenter=segmenter,
        video_dir=video_dir,
        selected_candidates=selected_candidates,
        start_obj_id=int(cfg["data"].get("ann_obj_id", 1)),
        merge_iou_thresh=float(tracking_cfg.get("merge_iou_thresh", 0.8)),
        merge_video_segments_fn=merge_video_segments_fn,
        torch_module=torch_module,
    )
    if not video_segments:
        print(f"[Skip] 未得到可用跟踪结果: {video_name}")
        return

    track_score = mean_selected_object_score(selected_candidates)
    save_tracking_results_fn(video_dir=video_dir, video_segments=video_segments, out_dir=out_dir)
    save_video_track_txt_fn(video_segments=video_segments, txt_path=track_txt_path, score=track_score)

    print(
        f"[Done] video={video_name} existing_categories={len(existing_categories)} "
        f"candidates={len(candidates)} sam_candidates={sam_candidate_count} "
        f"scored={len(scored_candidates)} "
        f"selected={len(selected_candidates)} tracked_frames={len(video_segments)} "
        f"score={track_score:.3f}"
    )
    for candidate in selected_candidates:
        print(
            f"[Info] key category={candidate.category} frame={candidate.frame_id} "
            f"range=[{candidate.start_frame_id},{candidate.end_frame_id}] "
            f"presence={candidate.category_presence_score:.3f} "
            f"det={candidate.det_conf:.3f} frame_score={candidate.frame_score:.3f} "
            f"sam_iou={candidate.sam_iou_score:.3f} object={candidate.object_score:.3f}"
        )
    print(f"[Done] 保存结果: {out_dir}")
    print(f"[Done] 轨迹文件: {track_txt_path}")


def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    runtime_cfg = cfg.get("runtime", {})
    cuda_visible_devices = runtime_cfg.get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    import torch
    from src.dataset_target_config import get_dataset_target_config
    from src.detector import DetectionResult
    from src.json_segmenter_and_tracker import JsonSegmenterAndTracker, merge_video_segments
    from utils.mask_to_box_track import save_video_track_txt
    from utils.visualization import save_tracking_results

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_target_config = get_dataset_target_config(cfg["data"]["dataset_name"])
    input_json_dir = Path(cfg["data"]["detector_json_root"])
    if not input_json_dir.is_dir():
        raise FileNotFoundError(f"Detector JSON directory does not exist: {input_json_dir}")
    json_paths = sorted(input_json_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No detector JSON files found in: {input_json_dir}")

    scores_by_video = load_video_scores(cfg["data"]["input_json"])
    if not scores_by_video:
        raise ValueError(f"No video scores loaded from: {cfg['data']['input_json']}")

    print(
        f"[Info] dataset={dataset_target_config.name} json_files={len(json_paths)} "
        f"score_videos={len(scores_by_video)} device={device}"
    )

    segmenter = JsonSegmenterAndTracker(
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
                scores_by_video=scores_by_video,
                cfg=cfg,
                dataset_target_config=dataset_target_config,
                segmenter=segmenter,
                detection_result_cls=DetectionResult,
                merge_video_segments_fn=merge_video_segments,
                save_tracking_results_fn=save_tracking_results,
                save_video_track_txt_fn=save_video_track_txt,
                torch_module=torch,
            )
        except Exception as exc:
            print(f"[Error] {json_path.name}: {exc}")


if __name__ == "__main__":
    main()
