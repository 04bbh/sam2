import argparse
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from dataclasses import dataclass
from typing import Dict, List

import torch
import numpy as np
import yaml

from utils.mask_to_box_track import save_video_track_txt
from src.dataset_target_config import get_dataset_target_config
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


def load_video_context(video_desc_root: str, video_path: str) -> str:
    if not video_desc_root:
        return ""

    desc_path = os.path.join(video_desc_root, os.path.basename(video_path) + ".json")
    if not os.path.isfile(desc_path):
        print(f"[Info] 未找到视频描述: {desc_path}")
        return ""

    try:
        with open(desc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Info] 视频描述读取失败: {desc_path} ({e})")
        return ""

    video_desc = data.get("video_desc", {})
    if not isinstance(video_desc, dict):
        print(f"[Info] 视频描述字段无效: {desc_path}")
        return ""

    video_context = video_desc.get("video_context", "")
    if not isinstance(video_context, str) or not video_context.strip():
        print(f"[Info] 视频描述为空: {desc_path}")
        return ""

    return video_context.strip()


def list_frames(video_dir: str) -> List[str]:
    names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    names.sort(key=lambda n: int(os.path.splitext(n)[0]))
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
    target_desc: str,
) -> None:
    videos_root = cfg["data"]["videos_root"]
    video_dir = os.path.join(videos_root, task.video_path)
    out_vis = os.path.join(cfg["data"]["detector_stage2_vis_root"], task.video_path)
    out_json = os.path.join(cfg["data"]["detector_stage2_json_root"], os.path.basename(video_dir) + ".json")


    if not os.path.isdir(video_dir):
        print(f"[Skip] 视频目录不存在: {video_dir}")
        return
    if os.path.isdir(out_vis) and os.path.isfile(out_json):
        print(f"[Skip] 检测可视化结果和json文件已存在: {out_vis} {out_json}")
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
        stride=int(cfg.get("data", {}).get("candidate_stride", 5)),
    )

    if not candidate_indices:
        print(f"[Skip] 无有效候选帧: {task.video_path}")
        return

    print(
        f"[Info] video={task.video_path} total_frames={len(frame_paths)} "
        f"score_bins={len(task.vid_bin_scores)} selected_score_bins={len(score_indices)} "
        f"candidate_frames={len(candidate_indices)}"
    )

    # 单阶段检测：使用 dataset_name 对应的 target_desc 做 batch 定位。
    batch_size = int(cfg["pipeline"]["batch_size"])
    seg_results = []
    det_count = 0
    stage2_target_desc = target_desc
    visualize_stage2 = cfg.get("pipeline", {}).get("visualize_detector", True)
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
    video_context = load_video_context(
        cfg.get("data", {}).get("video_desc_root", ""),
        task.video_path,
    )
    if video_context:
        print(f"[Info] 已加载视频描述: {task.video_path}")
    all_det_results = []

    for st in range(0, len(candidate_indices), batch_size):
        batch_indices = candidate_indices[st: st + batch_size]
        batch_paths = [frame_paths[i] for i in batch_indices]
        det_batch = detector.infer_batch(
            batch_paths,
            batch_indices,
            target_desc=stage2_target_desc,
            video_context=video_context,
        )
        all_det_results.extend(det_batch)
        det_count += len(det_batch)
        if visualize_stage2:
            save_detection_results(det_batch, detector_stage2_vis_dir)

        # seg_results.extend(segmenter_and_tracker.segment_many(det_batch))

    save_detector_json(
        det_results=all_det_results,
        json_path=detector_stage2_json_path,
        target_desc=stage2_target_desc or "",
    )
    print(f"[Done] 检测 JSON: {detector_stage2_json_path}")

  


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--input_json", default="input_json/input_datas_ucf.json", type=str)
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    dataset_target_config = get_dataset_target_config(cfg["data"]["dataset_name"])
    tasks = load_tasks(args.input_json)
    if not tasks:
        print(f"未读取到任务: {args.input_json}")
        return
    print(
        f"[Info] dataset={dataset_target_config.name} "
        f"targets={len(dataset_target_config.target_desc.split('，'))}"
    )

    detector = QwenVLDetector(
        model_path=cfg["qwen"]["model_path"],
        target_desc=dataset_target_config.target_desc,
        max_new_tokens=cfg["qwen"]["max_new_tokens"],
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
    for task in tasks:
        try:
            run_single_video(
                detector,
                segmenter_and_tracker,
                task,
                cfg,
                dataset_target_config.target_desc,
            )
        except Exception as e:
            print(f"[Error] {task.video_path}: {e}")

if __name__ == "__main__":
    main()
