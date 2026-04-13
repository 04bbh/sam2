from dataclasses import dataclass
from typing import Optional
from typing import Dict
import cv2
import numpy as np

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
    ) -> None:
        self.sam_model = build_sam2(config_file=model_cfg, ckpt_path=checkpoint, device=device)
        self.image_predictor = SAM2ImagePredictor(self.sam_model)
        self.video_predictor =  build_sam2_video_predictor(
            config_file=model_cfg,
            ckpt_path=checkpoint,
            device=device,
        )

        self.multimask_output = multimask_output

    def segment(self, det: DetectionResult) -> SegmentationResult:
        image_bgr = cv2.imread(det.frame_path)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {det.frame_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        # 约定：box 全 0 表示目标不存在，直接给 iou=0、空 mask
        if np.allclose(det.box_xyxy, np.zeros(4, dtype=np.float32)):
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return SegmentationResult(
                frame_idx=det.frame_idx,
                frame_path=det.frame_path,
                box_xyxy=det.box_xyxy,
                confidence=det.confidence,
                iou_score=0.0,
                mask=empty_mask,
            )

        self.image_predictor.set_image(image_rgb)

        masks, iou_preds, _ = self.image_predictor.predict(
            box=det.box_xyxy,
            multimask_output=self.multimask_output,
        )

        best_idx = int(np.argmax(iou_preds))
        best_iou = float(iou_preds[best_idx])
        best_mask = masks[best_idx] > 0.0

        return SegmentationResult(
            frame_idx=det.frame_idx,
            frame_path=det.frame_path,
            box_xyxy=det.box_xyxy,
            confidence=det.confidence,
            iou_score=best_iou,
            mask=best_mask.astype(np.uint8),
        )

    def track_from_mask(
        self,
        video_dir: str,
        ann_frame_idx: int,
        ann_obj_id: int,
        mask: np.ndarray,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        state = self.video_predictor.init_state(video_path=video_dir)

        self.video_predictor.add_new_mask(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            mask=mask,
        )

        video_segments: Dict[int, Dict[int, np.ndarray]] = {}
        num_frames = state["num_frames"]

        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            state,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=num_frames,
            reverse=False,
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            state,
            start_frame_idx=ann_frame_idx,
            max_frame_num_to_track=num_frames,
            reverse=True,
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.uint8)
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        self.video_predictor.reset_state(state)
        return video_segments
