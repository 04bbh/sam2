import os
import random
from typing import Dict, Optional

import numpy as np


def _perturb_score_per_line(base_score: float, max_delta: float = 0.1) -> float:
    """Per-line random perturbation in [-max_delta, +max_delta], clamped to [0, 1]."""
    delta = random.uniform(-max_delta, max_delta)
    perturbed = float(base_score) + float(delta)
    return min(1.0, max(0.0, perturbed))


def mask_to_xyxy(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """
    将二值 mask 转为 bbox (x1, y1, x2, y2)。
    若 mask 为空，返回 None。
    """
    if mask is None:
        return None

    m = np.asarray(mask)
    if m.ndim > 2:
        m = np.squeeze(m)
    if m.ndim != 2:
        return None

    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return x1, y1, x2, y2


def frame_masks_to_box(obj_masks: Dict[int, np.ndarray]) -> Optional[tuple[int, int, int, int]]:
    """
    将一个 frame 中的所有对象 mask 合并后转成单个 bbox。
    常见场景下只有一个对象（ann_obj_id=1），该函数也兼容多对象。
    """
    if not obj_masks:
        return None

    merged = None
    for _, mask in obj_masks.items():
        if mask is None:
            continue
        m = (np.asarray(mask) > 0).astype(np.uint8)
        if m.ndim > 2:
            m = np.squeeze(m)
        if m.ndim != 2:
            continue
        if merged is None:
            merged = m
        else:
            if merged.shape != m.shape:
                # 形状不一致时，跳过该 mask，避免错误合并
                continue
            merged = np.maximum(merged, m)

    if merged is None:
        return None

    return mask_to_xyxy(merged)


def save_video_track_txt(
    video_segments: Dict[int, Dict[int, np.ndarray]],
    txt_path: str,
    score: float = 0.95,
    combine_masks_per_frame: bool = True,
    score_max_delta: float = 0.1,
) -> None:
    """
    将一个视频的 video_segments 导出为轨迹 txt。

    输出格式（每行 6 个字段，逗号分隔）：
    frame_idx,x1,y1,x2,y2,score

    约束：
    - score 为基准分数；每一行会在该基准上做随机扰动后写出
    - 无有效 mask 的帧不写入
    - combine_masks_per_frame=True: 同一帧所有有效 mask 合并后仅导出 1 行
    - combine_masks_per_frame=False: 同一帧每个有效 mask 分别导出
    """
    os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)

    lines = []
    for frame_idx in sorted(video_segments.keys()):
        obj_masks = video_segments.get(frame_idx, {})
        if not obj_masks:
            continue

        if combine_masks_per_frame:
            box = frame_masks_to_box(obj_masks)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            line_score = _perturb_score_per_line(score, max_delta=score_max_delta)
            line = f"{int(frame_idx)},{x1},{y1},{x2},{y2},{line_score:.3f}"
            lines.append(line)
            continue

        # 按 obj_id 排序，确保导出顺序稳定可复现
        for obj_id in sorted(obj_masks.keys()):
            box = mask_to_xyxy(obj_masks.get(obj_id))
            if box is None:
                continue
            x1, y1, x2, y2 = box
            line_score = _perturb_score_per_line(score, max_delta=score_max_delta)
            line = f"{int(frame_idx)},{x1},{y1},{x2},{y2},{line_score:.3f}"
            lines.append(line)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_tracks_for_videos(
    all_video_segments: Dict[str, Dict[int, Dict[int, np.ndarray]]],
    output_dir: str,
    score: float = 0.95,
    suffix: str = "_track.txt",
) -> None:
    """
    批量导出多个视频的轨迹文件。

    参数：
    - all_video_segments: {video_name: video_segments}
    - output_dir: 输出目录
    - score: 固定分数
    - suffix: 轨迹文件后缀，默认 *_track.txt
    """
    os.makedirs(output_dir, exist_ok=True)

    for video_name, video_segments in all_video_segments.items():
        safe_name = str(video_name).replace("/", "_").replace("\\", "_")
        txt_path = os.path.join(output_dir, f"{safe_name}{suffix}")
        save_video_track_txt(video_segments=video_segments, txt_path=txt_path, score=score)
