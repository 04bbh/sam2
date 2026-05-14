from dataclasses import dataclass
from typing import Dict, List, Sequence
import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from src.detector import DetectionResult


@dataclass
class SegmentationResult:
    frame_idx: int
    frame_path: str
    box_xyxy: np.ndarray
    confidence: float
    iou_score: float
    mask: np.ndarray


class SegmenterAndTracker:

    def __init__(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str = "cuda",
        multimask_output: bool = False,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        async_loading_frames: bool = False,
    ) -> None:
        self.sam_model = build_sam2(config_file=model_cfg, ckpt_path=checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(self.sam_model)
        self.video_predictor =  build_sam2_video_predictor(
            config_file=model_cfg,
            ckpt_path=checkpoint,
            device=device,
        )

        self.multimask_output = multimask_output
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self.async_loading_frames = async_loading_frames

    def segment(self, det: DetectionResult) -> SegmentationResult:
        return self.segment_many([det])[0]

    @staticmethod
    def _is_empty_box(box_xyxy: np.ndarray) -> bool:
        return np.allclose(box_xyxy, np.zeros(4, dtype=np.float32))

    @staticmethod
    def _empty_result(det: DetectionResult, height: int, width: int) -> SegmentationResult:
        return SegmentationResult(
            frame_idx=det.frame_idx,
            frame_path=det.frame_path,
            box_xyxy=det.box_xyxy,
            confidence=det.confidence,
            iou_score=0.0,
            mask=np.zeros((height, width), dtype=np.uint8),
        )

    @staticmethod
    def _pick_best_masks(masks: np.ndarray, iou_preds: np.ndarray) -> List[tuple[np.ndarray, float]]:
        if masks.ndim == 3:
            masks = masks[None, ...]
        if iou_preds.ndim == 1:
            iou_preds = iou_preds[None, ...]

        best: List[tuple[np.ndarray, float]] = []
        for det_masks, det_ious in zip(masks, iou_preds):
            best_idx = int(np.argmax(det_ious))
            best.append(((det_masks[best_idx] > 0.0).astype(np.uint8), float(det_ious[best_idx])))
        return best

    def segment_many(self, detections: Sequence[DetectionResult]) -> List[SegmentationResult]:
        if not detections:
            return []

        results: List[SegmentationResult | None] = [None] * len(detections)
        grouped: Dict[str, List[tuple[int, DetectionResult]]] = {}
        for output_idx, det in enumerate(detections):
            grouped.setdefault(det.frame_path, []).append((output_idx, det))

        for frame_path, frame_detections in grouped.items():
            image_bgr = cv2.imread(frame_path)
            if image_bgr is None:
                raise FileNotFoundError(f"无法读取图像: {frame_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]

            valid_items = []
            for output_idx, det in frame_detections:
                if self._is_empty_box(det.box_xyxy):
                    results[output_idx] = self._empty_result(det, h, w)
                else:
                    valid_items.append((output_idx, det))

            if not valid_items:
                continue

            boxes = np.stack([det.box_xyxy for _, det in valid_items]).astype(np.float32)
            self.image_predictor.set_image(image_rgb)

            masks, iou_preds, _ = self.image_predictor.predict(
                box=boxes,
                multimask_output=self.multimask_output,
            )

            for (output_idx, det), (best_mask, best_iou) in zip(
                valid_items,
                self._pick_best_masks(masks, iou_preds),
            ):
                results[output_idx] = SegmentationResult(
                    frame_idx=det.frame_idx,
                    frame_path=det.frame_path,
                    box_xyxy=det.box_xyxy,
                    confidence=det.confidence,
                    iou_score=best_iou,
                    mask=best_mask,
                )

        return [res for res in results if res is not None]

    def track_from_mask(
        self,
        video_dir: str,
        ann_frame_idx: int,
        ann_obj_id: int,
        mask: np.ndarray,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        return self.track_from_masks(
            video_dir=video_dir,
            ann_frame_idx=ann_frame_idx,
            obj_ids=[ann_obj_id],
            masks=[mask],
        )

    def track_from_masks(
        self,
        video_dir: str,
        ann_frame_idx: int,
        obj_ids: Sequence[int],
        masks: Sequence[np.ndarray],
        start_frame_idx: int | None = None,
        end_frame_idx: int | None = None,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        if len(obj_ids) != len(masks):
            raise ValueError("obj_ids 与 masks 长度不一致")
        if not masks:
            return {}

        state = self.video_predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=self.async_loading_frames,
        )

        try:
            for obj_id, mask in zip(obj_ids, masks):
                self.video_predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=ann_frame_idx,
                    obj_id=int(obj_id),
                    mask=mask,
                )

            video_segments: Dict[int, Dict[int, np.ndarray]] = {}
            num_frames = state["num_frames"]
            range_start = ann_frame_idx if start_frame_idx is None else int(start_frame_idx)
            range_end = (num_frames - 1) if end_frame_idx is None else int(end_frame_idx)
            range_start = max(0, range_start)
            range_end = min(num_frames - 1, range_end)
            if range_end < range_start:
                return {}

            forward_frames = max(0, range_end - ann_frame_idx)
            backward_frames = max(0, ann_frame_idx - range_start)

            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                state,
                start_frame_idx=ann_frame_idx,
                max_frame_num_to_track=forward_frames,
                reverse=False,
            ):
                if out_frame_idx < range_start or out_frame_idx > range_end:
                    continue
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                state,
                start_frame_idx=ann_frame_idx,
                max_frame_num_to_track=backward_frames,
                reverse=True,
            ):
                if out_frame_idx < range_start or out_frame_idx > range_end:
                    continue
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            return video_segments
        finally:
            self.video_predictor.reset_state(state)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
