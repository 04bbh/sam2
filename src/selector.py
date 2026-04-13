from dataclasses import dataclass
from typing import Iterable, List, Optional

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


class FrameSelector:
    def __init__(self, alpha: float, beta: float) -> None:
        if alpha < 0 or beta < 0:
            raise ValueError("alpha/beta 必须为非负数")
        if alpha + beta == 0:
            raise ValueError("alpha+beta 不能为 0")
        self.alpha = alpha
        self.beta = beta

    def score(self, seg: SegmentationResult) -> ScoredFrame:
        score = self.alpha * seg.confidence + self.beta * seg.iou_score
        return ScoredFrame(
            frame_idx=seg.frame_idx,
            frame_path=seg.frame_path,
            score=score,
            confidence=seg.confidence,
            iou_score=seg.iou_score,
            mask=seg.mask,
            box_xyxy=seg.box_xyxy,
        )

    def select_best(self, seg_results: Iterable[SegmentationResult]) -> Optional[ScoredFrame]:
        scored: List[ScoredFrame] = [self.score(seg) for seg in seg_results]
        if not scored:
            raise ValueError("没有可用于排序的分割结果")

        best = max(scored, key=lambda s: s.score)
        if best.score < 0.8:
            return None
        return best
