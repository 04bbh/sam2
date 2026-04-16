import argparse
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,0,1"

from dataclasses import dataclass
from typing import Dict, List

import torch
import numpy as np
import yaml

from utils.mask_to_box_track import save_video_track_txt
from src.detector import QwenVLDetector
from src.segmenter_and_tracker import SegmenterAndTracker
from src.selector import FrameSelector
from utils.visualization import save_detection_results, save_tracking_results



device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class VideoTask:
    video_path: str
    vid_bin_scores: List[float]


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_tasks(json_path: str) -> List[VideoTask]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks: List[VideoTask] = []
    for item in data:
        video_path = item.get("video_path", "")
        scores = item.get("vid_bin_scores", [])
        if not video_path:
            continue
        if not isinstance(scores, list):
            scores = []

        parsed_scores: List[float] = []
        for s in scores:
            try:
                parsed_scores.append(float(s))
            except Exception:
                parsed_scores.append(0.0)

        tasks.append(VideoTask(video_path=video_path, vid_bin_scores=parsed_scores))

    return tasks


def list_frames(video_dir: str) -> List[str]:
    names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    names.sort()
    return [os.path.join(video_dir, n) for n in names]


def select_stratified_score_indices(
    scores: List[float],
    high_ratio: float,
    mid_ratio: float,
    low_ratio: float,
    uniform_interval: int,
) -> List[int]:
    """
    对 vid_bin_scores 做分数分层采样，并额外按时间间隔补采样。

    采样方式：
    - 按分数从高到低排序后划分为高/中/低三层
    - 高分层取 high_ratio
    - 中分层取 mid_ratio
    - 低分层取 low_ratio
    - 再每隔 uniform_interval 个 score bin 补一个候选，避免时间段完全空白
    """
    if not scores:
        return []
    if min(high_ratio, mid_ratio, low_ratio) < 0:
        raise ValueError("采样比例必须 >= 0")

    n = len(scores)
    ranked = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
    layer_size = int(math.ceil(n / 3.0))
    layers = [
        (ranked[:layer_size], high_ratio),
        (ranked[layer_size: layer_size * 2], mid_ratio),
        (ranked[layer_size * 2:], low_ratio),
    ]

    selected = set()
    for layer, ratio in layers:
        if not layer or ratio <= 0:
            continue
        k = max(1, int(math.ceil(len(layer) * ratio)))
        selected.update(idx for idx, _ in layer[:k])

    if uniform_interval > 0:
        selected.update(range(0, n, uniform_interval))

    return sorted(selected)


def pick_candidate_indices(total_frames: int, score_indices: List[int], stride: int = 5) -> List[int]:
    """
    将分数索引映射到视频帧索引：frame_idx = score_idx * stride。
    并过滤到有效范围内。
    """
    mapped = [idx * stride for idx in score_indices]
    valid = sorted({idx for idx in mapped if 0 <= idx < total_frames})
    return valid


def merge_video_segments(target: dict, source: dict, iou_thresh: float = 0.8) -> None:
    def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
        a_bool = a.astype(bool)
        b_bool = b.astype(bool)
        inter = np.logical_and(a_bool, b_bool).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a_bool, b_bool).sum()
        return float(inter) / float(union) if union > 0 else 0.0

    for frame_idx, obj_masks in source.items():
        merged = {}
        if frame_idx in target:
            merged.update(target[frame_idx])
        merged.update(obj_masks)

        items = []
        for obj_id, mask in merged.items():
            if mask is None:
                continue
            m = np.asarray(mask)
            if m.ndim > 2:
                m = np.squeeze(m)
            if m.ndim != 2:
                continue
            area = int((m > 0).sum())
            if area == 0:
                continue
            items.append((obj_id, m.astype(np.uint8), area))

        items.sort(key=lambda x: x[2], reverse=True)
        kept: list[tuple[int, np.ndarray]] = []
        for obj_id, mask, _ in items:
            if all(mask_iou(mask, kept_mask) <= iou_thresh for _, kept_mask in kept):
                kept.append((obj_id, mask))

        target[frame_idx] = {obj_id: mask for obj_id, mask in kept}


def save_detector_json(
    det_results: List,
    json_path: str,
    target_desc: str,
) -> None:
    grouped: Dict[int, dict] = {}
    for det in det_results:
        frame_item = grouped.setdefault(
            det.frame_idx,
            {
                "frame_id": int(det.frame_idx),
                "target_desc": target_desc,
                "detections": [],
            },
        )

        box = np.asarray(det.box_xyxy, dtype=float).tolist()
        is_empty = det.confidence <= 0 and not det.category and np.allclose(det.box_xyxy, 0)
        if is_empty:
            continue

        frame_item["detections"].append(
            {
                "category": det.category,
                "box": [float(x) for x in box],
                "confidence": float(det.confidence),
            }
        )

    data = [grouped[frame_idx] for frame_idx in sorted(grouped)]
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_single_video(
    detector: QwenVLDetector,
    segmenter_and_tracker: SegmenterAndTracker,
    task: VideoTask,
    cfg: dict,
    log_file: str | None = None,
) -> None:
    videos_root = cfg["data"]["videos_root"]
    video_dir = os.path.join(videos_root, task.video_path)
    out_dir = os.path.join(cfg["data"]["output_root"], task.video_path)
    track_txt_path = os.path.join(cfg["data"]["track_txt_root"], os.path.basename(video_dir) + ".txt")


    if not os.path.isdir(video_dir):
        print(f"[Skip] 视频目录不存在: {video_dir}")
        return
    if os.path.isdir(out_dir) and os.path.isfile(track_txt_path):
        print(f"[Skip] 输出目录和轨迹文件已存在: {out_dir} {track_txt_path}")
        return

    frame_paths = list_frames(video_dir)
    if not frame_paths:
        print(f"[Skip] 无帧图像: {video_dir}")
        return

    candidate_cfg = cfg.get("candidate_selector", {})
    score_indices = select_stratified_score_indices(
        scores=task.vid_bin_scores,
        high_ratio=candidate_cfg.get("high_ratio", 0.30),
        mid_ratio=candidate_cfg.get("mid_ratio", 0.15),
        low_ratio=candidate_cfg.get("low_ratio", 0.05),
        uniform_interval=int(candidate_cfg.get("uniform_interval", 10)),
    )
    candidate_indices = pick_candidate_indices(
        total_frames=len(frame_paths),
        score_indices=score_indices,
        stride=5,
    )

    if not candidate_indices:
        print(f"[Skip] 无有效候选帧: {task.video_path}")
        return

    print(
        f"[Info] video={task.video_path} total_frames={len(frame_paths)} "
        f"score_bins={len(task.vid_bin_scores)} selected_score_bins={len(score_indices)} "
        f"candidate_frames={len(candidate_indices)}"
    )

    # 两阶段检测：
    # 第一阶段：对候选帧整体判断异常类别（可多类）。
    # 第二阶段：分 batch 定位，使用第一阶段输出的类别作为目标类别。
    batch_size = int(cfg["pipeline"]["batch_size"])
    seg_results = []
    det_count = 0
    stage2_target_desc = None if detector.use_two_stage else cfg["qwen"]["target_desc"]
    visualize_stage2 = cfg.get("pipeline", {}).get("visualize_detector_stage2", True)
    detector_stage2_vis_root = cfg.get("data", {}).get(
        "detector_stage2_vis_root",
        "./outputs/detector_stage2_vis",
    )
    detector_stage2_vis_dir = os.path.join(detector_stage2_vis_root, task.video_path)
    detector_stage2_json_root = cfg.get("data", {}).get(
        "detector_stage2_json_root",
        "./outputs/detector_stage2_json",
    )
    detector_stage2_json_path = os.path.join(detector_stage2_json_root, task.video_path + ".json")
    all_det_results = []

    if detector.use_two_stage:
        stage1_paths = [frame_paths[i] for i in candidate_indices]
        stage1_items = detector.detect_categories_with_reason(stage1_paths)
        categories = []
        for item in stage1_items:
            name = item.get("name")
            if isinstance(name, str) and name not in categories:
                categories.append(name)
        if not categories:
            print(f"[Skip] 未识别到目标类别: {task.video_path}")
            return
        stage2_target_desc = "，".join(categories)
        print(f"[Stage1] video={task.video_path} categories={categories} reasons={stage1_items}")

    for st in range(0, len(candidate_indices), batch_size):
        batch_indices = candidate_indices[st: st + batch_size]
        batch_paths = [frame_paths[i] for i in batch_indices]
        det_batch = detector.infer_batch(batch_paths, batch_indices, target_desc=stage2_target_desc)
        all_det_results.extend(det_batch)
        det_count += len(det_batch)
        if log_file is not None:
            log_file.write(str(det_batch))
            log_file.write('\n')
        if visualize_stage2:
            save_detection_results(det_batch, detector_stage2_vis_dir)

        # seg_results.extend(segmenter_and_tracker.segment_many(det_batch))

    save_detector_json(
        det_results=all_det_results,
        json_path=detector_stage2_json_path,
        target_desc=stage2_target_desc or "",
    )
    print(f"[Done] 检测 JSON: {detector_stage2_json_path}")

    # if not seg_results:
    #     print(f"[Skip] 未得到可用分割结果: {task.video_path}")
    #     return

    # valid_seg_count = sum(1 for seg in seg_results if not (seg.box_xyxy == 0).all())
    # print(
    #     f"[Detect] video={task.video_path} detections={det_count} "
    #     f"segmentations={len(seg_results)} valid_segmentations={valid_seg_count}"
    # )

    # # 在候选帧中筛选关键帧
    # selector_cfg = cfg["selector"]
    # selector = FrameSelector(
    #     alpha=selector_cfg["alpha"],
    #     beta=selector_cfg["beta"],
    #     min_score=selector_cfg.get("min_score", 0.6),
    #     count_bonus=selector_cfg.get("count_bonus", 0.03),
    #     max_targets_per_frame=selector_cfg.get("max_targets_per_frame", 3),
    #     min_frame_gap=selector_cfg.get("min_frame_gap", 30),
    # )
    # top_k_frames = int(selector_cfg.get("top_k_frames", 3))
    # best_frames = selector.select_top_frames(seg_results, top_k=top_k_frames)
    # if not best_frames:
    #     print(f"[Skip] 关键帧得分均低于阈值: {task.video_path}")
    #     return

    # for rank, frame in enumerate(best_frames, start=1):
    #     print(
    #         f"[Best-{rank}] video={task.video_path} frame_idx={frame.frame_idx} "
    #         f"score={frame.score:.4f} targets={len(frame.segmentations)} "
    #         f"(avg_qwen_conf={frame.confidence:.4f}, avg_sam_iou={frame.iou_score:.4f})"
    #         f"segmentations={frame.segmentations}"
    #     )

    # # 同一个关键帧里的多个目标一次性加入同一个 SAM2 state，再统一传播。
    # video_segments = {}
    # next_obj_id = int(cfg["data"]["ann_obj_id"])
    # track_count = 0
    # merge_iou_thresh = selector_cfg.get("merge_iou_thresh", 0.8)
    # for frame in best_frames:
    #     obj_ids = list(range(next_obj_id, next_obj_id + len(frame.segmentations)))
    #     masks = [seg.mask for seg in frame.segmentations]
    #     tracked_segments = segmenter_and_tracker.track_from_masks(
    #         video_dir=video_dir,
    #         ann_frame_idx=frame.frame_idx,
    #         obj_ids=obj_ids,
    #         masks=masks,
    #     )
    #     merge_video_segments(video_segments, tracked_segments, iou_thresh=merge_iou_thresh)
    #     next_obj_id += len(frame.segmentations)
    #     track_count += len(frame.segmentations)

    # if not video_segments:
    #     print(f"[Skip] 未得到可用跟踪结果: {task.video_path}")
    #     return

    # print(
    #     f"[Track] video={task.video_path} keyframes={len(best_frames)} "
    #     f"tracks={track_count} tracked_frames={len(video_segments)}"
    # )

    
    # save_tracking_results(video_dir=video_dir, video_segments=video_segments, out_dir=out_dir)

    # save_video_track_txt(video_segments=video_segments, txt_path=track_txt_path, score=1)

    # print(f"[Done] 保存结果: {out_dir}")
    # print(f"[Done] 轨迹文件: {track_txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--input_json", default="sht_json/input_datas_sht_original_scaled_chaserunmove.json", type=str)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    tasks = load_tasks(args.input_json)
    if not tasks:
        print(f"未读取到任务: {args.input_json}")
        return

    detector = QwenVLDetector(
        model_path=cfg["qwen"]["model_path"],
        target_desc=cfg["qwen"]["target_desc"],
        max_new_tokens=cfg["qwen"]["max_new_tokens"],
        stage1_max_new_tokens=cfg["qwen"].get("stage1_max_new_tokens", 128),
        use_two_stage=cfg["qwen"].get("use_two_stage", False),
        use_quantization=False,
        use_flashattn=True,
        device=device,
        batch_size=cfg["pipeline"]["batch_size"],
    )

    segmenter_and_tracker = SegmenterAndTracker(
        model_cfg=cfg["sam2"]["model_cfg"],
        checkpoint=cfg["sam2"]["checkpoint"],
        multimask_output=cfg["sam2"]["multimask_output"],
        device=device,
    )
    log_file = open('logs/detector_stage2_logs/log1.txt', 'w')
    for task in tasks:
        try:
            run_single_video(detector, segmenter_and_tracker, task, cfg, log_file)
        except Exception as e:
            print(f"[Error] {task.video_path}: {e}")
    log_file.close()

if __name__ == "__main__":
    main()
