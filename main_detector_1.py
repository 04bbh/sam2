import argparse
import json
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,0,1"

from dataclasses import dataclass
from typing import List

import torch
import yaml

from utils.mask_to_box_track import save_video_track_txt
from src.detector import QwenVLDetector
from src.segmenter_and_tracker import SegmenterAndTracker
from src.selector import FrameSelector
from utils.visualization import save_tracking_results



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


def select_top_25_percent_indices(scores: List[float]) -> List[int]:
    """
    选取 vid_bin_scores 中 Top-25% 分数对应的索引。
    至少返回 1 个索引（如果 scores 非空）。
    """
    if not scores:
        return []

    n = len(scores)
    k = max(1, int(math.ceil(n * 0.25)))

    # 按分数从高到低排序；分数相同按索引从小到大
    ranked = sorted(enumerate(scores), key=lambda x: (-x[1], x[0]))
    top_indices = [idx for idx, _ in ranked[:k]]
    top_indices.sort()
    return top_indices


def pick_candidate_indices(total_frames: int, score_indices: List[int], stride: int = 5) -> List[int]:
    """
    将分数索引映射到视频帧索引：frame_idx = score_idx * stride。
    并过滤到有效范围内。
    """
    mapped = [idx * stride for idx in score_indices]
    valid = sorted({idx for idx in mapped if 0 <= idx < total_frames})
    return valid


def run_single_video(
    detector: QwenVLDetector,
    segmenter_and_tracker: SegmenterAndTracker,
    task: VideoTask,
    cfg: dict,
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

    top_score_indices = select_top_25_percent_indices(task.vid_bin_scores)
    candidate_indices = pick_candidate_indices(
        total_frames=len(frame_paths),
        score_indices=top_score_indices,
        stride=5,
    )

    if not candidate_indices:
        print(f"[Skip] 无有效候选帧: {task.video_path}")
        return

    print(
        f"[Info] video={task.video_path} total_frames={len(frame_paths)} "
        f"top25_score_bins={len(top_score_indices)} candidate_frames={len(candidate_indices)}"
    )

    # 只在候选帧（score_idx*5）上做检测+分割
    batch_size = int(cfg["pipeline"]["batch_size"])
    seg_results = []
    for st in range(0, len(candidate_indices), batch_size):
        batch_indices = candidate_indices[st: st + batch_size]
        batch_paths = [frame_paths[i] for i in batch_indices]

        det_batch = detector.infer_batch(batch_paths, batch_indices)
        for det in det_batch:
            seg_results.append(segmenter_and_tracker.segment(det))

    if not seg_results:
        print(f"[Skip] 未得到可用分割结果: {task.video_path}")
        return

    # 在候选帧中筛选关键帧
    selector = FrameSelector(alpha=cfg["selector"]["alpha"], beta=cfg["selector"]["beta"])
    best = selector.select_best(seg_results)
    if best is None:
        print(f"[Skip] 关键帧得分均低于阈值: {task.video_path}")
        return

    print(
        f"[Best] video={task.video_path} frame_idx={best.frame_idx} "
        f"score={best.score:.4f} (qwen_conf={best.confidence:.4f}, sam_iou={best.iou_score:.4f})"
    )

    # 从关键帧向全视频传播
    video_segments = segmenter_and_tracker.track_from_mask(
        video_dir=video_dir,
        ann_frame_idx=best.frame_idx,
        ann_obj_id=int(cfg["data"]["ann_obj_id"]),
        mask=best.mask,
    )

    
    save_tracking_results(video_dir=video_dir, video_segments=video_segments, out_dir=out_dir)

    save_video_track_txt(video_segments=video_segments, txt_path=track_txt_path, score=1)

    print(f"[Done] 保存结果: {out_dir}")
    print(f"[Done] 轨迹文件: {track_txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    parser.add_argument("--input_json", default="sht_json/input_datas_sht_original_scaled.json", type=str)
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
            run_single_video(detector, segmenter_and_tracker, task, cfg)
        except Exception as e:
            print(f"[Error] {task.video_path}: {e}")


if __name__ == "__main__":
    main()
