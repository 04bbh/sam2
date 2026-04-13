import os
import re
from pathlib import Path
from typing import List

import cv2


FRAMES_ROOT = Path("/data/users/wjq/datasets/shanghaitech/testing/frames")
VIDEOS_ROOT = Path("/data/users/wjq/datasets/shanghaitech/testing/videos")
FPS = 24


def numeric_sort_key(path: Path):
    """按文件名中的数字顺序排序，兼容 1.jpg/001.jpg/000001.jpg。"""
    stem = path.stem
    m = re.search(r"\d+", stem)
    if m:
        return int(m.group())
    return stem


def list_frame_paths(video_frame_dir: Path) -> List[Path]:
    frames = [
        p
        for p in video_frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    frames.sort(key=numeric_sort_key)
    return frames


def write_video_from_frames(frame_paths: List[Path], out_video_path: Path, fps: int = FPS) -> int:
    """将一组帧写成 mp4 视频，返回实际写入帧数。"""
    if not frame_paths:
        return 0

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        print(f"[Skip] 无法读取首帧: {frame_paths[0]}")
        return 0

    h, w = first.shape[:2]
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4 编码，优先使用 mp4v
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, float(fps), (w, h))

    if not writer.isOpened():
        print(f"[Skip] 无法创建视频文件: {out_video_path}")
        return 0

    written = 0
    for p in frame_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[Warn] 跳过损坏帧: {p}")
            continue

        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        writer.write(img)
        written += 1

    writer.release()
    return written


def main() -> None:
    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"帧目录不存在: {FRAMES_ROOT}")

    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted([p for p in FRAMES_ROOT.iterdir() if p.is_dir()])
    if not video_dirs:
        print(f"未找到视频帧子目录: {FRAMES_ROOT}")
        return

    print(f"发现视频目录数量: {len(video_dirs)}")

    total_written = 0
    for video_dir in video_dirs:
        frame_paths = list_frame_paths(video_dir)
        if not frame_paths:
            print(f"[Skip] 无可用帧: {video_dir}")
            continue

        out_path = VIDEOS_ROOT / f"{video_dir.name}.mp4"
        cnt = write_video_from_frames(frame_paths, out_path, fps=FPS)

        if cnt == 0:
            print(f"[Skip] 合成失败: {video_dir.name}")
            continue

        total_written += cnt
        print(f"[Done] {video_dir.name} -> {out_path} ({cnt} 帧, {FPS} FPS)")

    print(f"全部完成，总写入帧数: {total_written}")


if __name__ == "__main__":
    main()
