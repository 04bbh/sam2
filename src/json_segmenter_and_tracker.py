from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np

from src.detector import DetectionResult
from src.segmenter_and_tracker import SegmentationResult, SegmenterAndTracker


def list_frames(video_dir: str) -> List[str]:
    names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    names.sort()
    return [os.path.join(video_dir, name) for name in names]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a).astype(bool)
    b_bool = np.asarray(b).astype(bool)
    inter = np.logical_and(a_bool, b_bool).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a_bool, b_bool).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def merge_video_segments(
    target: Dict[int, Dict[int, np.ndarray]],
    source: Dict[int, Dict[int, np.ndarray]],
    iou_thresh: float = 0.8,
) -> None:
    for frame_idx, obj_masks in source.items():
        merged: Dict[int, np.ndarray] = {}
        if frame_idx in target:
            merged.update(target[frame_idx])
        merged.update(obj_masks)

        items: list[tuple[int, np.ndarray, int]] = []
        for obj_id, mask in merged.items():
            if mask is None:
                continue
            m = np.asarray(mask)
            if m.ndim > 2:
                m = np.squeeze(m)
            if m.ndim != 2:
                continue
            mask_uint8 = (m > 0).astype(np.uint8)
            area = int(mask_uint8.sum())
            if area == 0:
                continue
            items.append((int(obj_id), mask_uint8, area))

        items.sort(key=lambda x: x[2], reverse=True)
        kept: list[tuple[int, np.ndarray]] = []
        for obj_id, mask, _ in items:
            if all(mask_iou(mask, kept_mask) <= iou_thresh for _, kept_mask in kept):
                kept.append((obj_id, mask))

        target[frame_idx] = {obj_id: mask for obj_id, mask in kept}


class JsonSegmenterAndTracker(SegmenterAndTracker):
    @staticmethod
    def _parse_box(box: Any) -> np.ndarray:
        if not isinstance(box, list) or len(box) != 4:
            return np.zeros(4, dtype=np.float32)
        try:
            box_arr = np.asarray(box, dtype=np.float32)
        except Exception:
            return np.zeros(4, dtype=np.float32)
        if not np.isfinite(box_arr).all():
            return np.zeros(4, dtype=np.float32)
        return box_arr

    @staticmethod
    def _parse_confidence(detection: dict[str, Any]) -> float:
        try:
            confidence = float(detection.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _parse_category(detection: dict[str, Any]) -> str:
        category = detection.get("category", detection.get("name", ""))
        if not isinstance(category, str):
            return ""
        return category.strip()

    @staticmethod
    def json_blocks_to_detections(
        video_dir: str,
        blocks: list[dict[str, Any]],
    ) -> List[DetectionResult]:
        frame_paths = list_frames(video_dir)
        detections: List[DetectionResult] = []

        for block in blocks:
            try:
                frame_idx = int(block.get("frame_id", block.get("frame_idx", -1)))
            except Exception:
                frame_idx = -1
            if frame_idx < 0 or frame_idx >= len(frame_paths):
                continue

            block_detections = block.get("detections", [])
            if not isinstance(block_detections, list):
                continue

            for detection in block_detections:
                if not isinstance(detection, dict):
                    continue
                box_xyxy = JsonSegmenterAndTracker._parse_box(detection.get("box"))
                if np.allclose(box_xyxy, np.zeros(4, dtype=np.float32)):
                    continue
                detections.append(
                    DetectionResult(
                        frame_idx=frame_idx,
                        frame_path=frame_paths[frame_idx],
                        box_xyxy=box_xyxy,
                        confidence=JsonSegmenterAndTracker._parse_confidence(detection),
                        category=JsonSegmenterAndTracker._parse_category(detection),
                    )
                )

        return detections

    def segment_and_track_from_json(
        self,
        video_dir: str,
        blocks: list[dict[str, Any]],
        start_obj_id: int,
        merge_iou_thresh: float,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        detections = self.json_blocks_to_detections(video_dir=video_dir, blocks=blocks)
        if not detections:
            return {}

        segmentations = self.segment_many(detections)
        valid_segmentations = [
            seg
            for seg in segmentations
            if not np.allclose(seg.box_xyxy, np.zeros(4, dtype=np.float32))
            and np.asarray(seg.mask).sum() > 0
        ]
        if not valid_segmentations:
            return {}

        grouped: Dict[int, list] = {}
        for seg in valid_segmentations:
            grouped.setdefault(int(seg.frame_idx), []).append(seg)

        video_segments: Dict[int, Dict[int, np.ndarray]] = {}
        next_obj_id = int(start_obj_id)

        for frame_idx in sorted(grouped):
            frame_segmentations = grouped[frame_idx]
            obj_ids = list(range(next_obj_id, next_obj_id + len(frame_segmentations)))
            masks = [seg.mask for seg in frame_segmentations]
            tracked_segments = self.track_from_masks(
                video_dir=video_dir,
                ann_frame_idx=frame_idx,
                obj_ids=obj_ids,
                masks=masks,
            )
            merge_video_segments(
                target=video_segments,
                source=tracked_segments,
                iou_thresh=merge_iou_thresh,
            )
            next_obj_id += len(frame_segmentations)

        return video_segments

    def segment_and_track_best_iou(
        self,
        video_dir: str,
        blocks: list[dict[str, Any]],
        start_obj_id: int,
        merge_iou_thresh: float,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        detections = self.json_blocks_to_detections(video_dir=video_dir, blocks=blocks)
        if not detections:
            return {}

        segmentations = self.segment_many(detections)
        candidates = []
        for order, (det, seg) in enumerate(zip(detections, segmentations)):
            category = det.category.strip()
            if not category:
                continue
            if np.allclose(seg.box_xyxy, np.zeros(4, dtype=np.float32)):
                continue
            if np.asarray(seg.mask).sum() <= 0:
                continue
            candidates.append((order, category, det, seg))
        if not candidates:
            return {}

        best_by_category: Dict[str, tuple[int, DetectionResult, SegmentationResult]] = {}
        for order, category, det, seg in candidates:
            current = best_by_category.get(category)
            candidate_key = (-float(seg.iou_score), -float(det.confidence), int(seg.frame_idx), order)
            if current is None:
                best_by_category[category] = (order, det, seg)
                continue

            current_order, current_det, current_seg = current
            current_key = (
                -float(current_seg.iou_score),
                -float(current_det.confidence),
                int(current_seg.frame_idx),
                current_order,
            )
            if candidate_key < current_key:
                best_by_category[category] = (order, det, seg)

        selected = [
            (category, order, det, seg)
            for category, (order, det, seg) in best_by_category.items()
        ]
        selected.sort(key=lambda item: (int(item[3].frame_idx), item[0], item[1]))

        grouped: Dict[int, list[tuple[str, int, DetectionResult, SegmentationResult]]] = {}
        for item in selected:
            grouped.setdefault(int(item[3].frame_idx), []).append(item)

        video_segments: Dict[int, Dict[int, np.ndarray]] = {}
        next_obj_id = int(start_obj_id)

        for frame_idx in sorted(grouped):
            frame_items = grouped[frame_idx]
            obj_ids = list(range(next_obj_id, next_obj_id + len(frame_items)))
            masks = [seg.mask for _, _, _, seg in frame_items]
            tracked_segments = self.track_from_masks(
                video_dir=video_dir,
                ann_frame_idx=frame_idx,
                obj_ids=obj_ids,
                masks=masks,
            )
            merge_video_segments(
                target=video_segments,
                source=tracked_segments,
                iou_thresh=merge_iou_thresh,
            )
            next_obj_id += len(frame_items)

        return video_segments


__all__ = [
    "JsonSegmenterAndTracker",
    "list_frames",
    "merge_video_segments",
]
