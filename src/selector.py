from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
from src.segmenter_and_tracker import SegmentationResult


@dataclass
class ScoredFrame:
    frame_idx: int
    frame_path: str
    score: float
    confidence: float
    iou_score: float
    mask: object
    box_xyxy: object
    segmentations: List[SegmentationResult] = field(default_factory=list)


class FrameSelector:
    def __init__(
        self,
        alpha: float,
        beta: float,
        min_score: float = 0.6,
        count_bonus: float = 0.03,
        max_targets_per_frame: int = 5,
        min_frame_gap: int = 0,
    ) -> None:
        if alpha < 0 or beta < 0:
            raise ValueError("alpha/beta 必须为非负数")
        if alpha + beta == 0:
            raise ValueError("alpha+beta 不能为 0")
        if max_targets_per_frame <= 0:
            raise ValueError("max_targets_per_frame 必须 > 0")
        if min_frame_gap < 0:
            raise ValueError("min_frame_gap 必须 >= 0")
        self.alpha = alpha
        self.beta = beta
        self.min_score = min_score
        self.count_bonus = count_bonus
        self.max_targets_per_frame = max_targets_per_frame
        self.min_frame_gap = min_frame_gap

    @staticmethod
    def _is_valid_segmentation(seg: SegmentationResult) -> bool:
        return not np.allclose(seg.box_xyxy, np.zeros(4, dtype=np.float32))

    def target_score(self, seg: SegmentationResult) -> float:
        return self.alpha * seg.confidence + self.beta * seg.iou_score

    def score(self, seg: SegmentationResult) -> ScoredFrame:
        score = self.target_score(seg)
        return ScoredFrame(
            frame_idx=seg.frame_idx,
            frame_path=seg.frame_path,
            score=score,
            confidence=seg.confidence,
            iou_score=seg.iou_score,
            mask=seg.mask,
            box_xyxy=seg.box_xyxy,
            segmentations=[seg] if self._is_valid_segmentation(seg) else [],
        )

    def score_frame(self, frame_segmentations: List[SegmentationResult]) -> ScoredFrame:
        if not frame_segmentations:
            raise ValueError("没有可用于排序的分割结果")

        valid = [seg for seg in frame_segmentations if self._is_valid_segmentation(seg)]
        if not valid:
            base = frame_segmentations[0]
            return ScoredFrame(
                frame_idx=base.frame_idx,
                frame_path=base.frame_path,
                score=0.0,
                confidence=0.0,
                iou_score=0.0,
                mask=base.mask,
                box_xyxy=base.box_xyxy,
                segmentations=[],
            )

        ranked = sorted(valid, key=self.target_score, reverse=True)
        top_targets = ranked[: self.max_targets_per_frame]
        target_scores = [self.target_score(seg) for seg in top_targets]
        mean_score = sum(target_scores) / len(target_scores)
        # score = min(1.0, mean_score + self.count_bonus * min(len(valid), self.max_targets_per_frame))
        score = mean_score
        confidence = sum(seg.confidence for seg in top_targets) / len(top_targets)
        iou_score = sum(seg.iou_score for seg in top_targets) / len(top_targets)
        best_target = top_targets[0]

        return ScoredFrame(
            frame_idx=best_target.frame_idx,
            frame_path=best_target.frame_path,
            score=score,
            confidence=confidence,
            iou_score=iou_score,
            mask=best_target.mask,
            box_xyxy=best_target.box_xyxy,
            segmentations=top_targets,
        )

    def select_top_frames(
        self,
        seg_results: Iterable[SegmentationResult],
        top_k: int = 1,
        min_score: Optional[float] = None,
    ) -> List[ScoredFrame]:
        if top_k <= 0:
            raise ValueError("top_k 必须 > 0")

        grouped: Dict[int, List[SegmentationResult]] = {}
        for seg in seg_results:
            grouped.setdefault(seg.frame_idx, []).append(seg)

        scored: List[ScoredFrame] = [self.score_frame(items) for items in grouped.values()]
        if not scored:
            raise ValueError("没有可用于排序的分割结果")

        threshold = self.min_score if min_score is None else min_score
        scored = [frame for frame in scored if frame.score >= threshold and frame.segmentations]
        scored.sort(key=lambda s: (-s.score, s.frame_idx))

        selected: List[ScoredFrame] = []
        for frame in scored:
            if all(abs(frame.frame_idx - chosen.frame_idx) >= self.min_frame_gap for chosen in selected):
                selected.append(frame)
            if len(selected) >= top_k:
                break
        return selected

    def select_best(self, seg_results: Iterable[SegmentationResult]) -> Optional[ScoredFrame]:
        selected = self.select_top_frames(seg_results, top_k=1)
        if not selected:
            return None
        return selected[0]
