import os
from pathlib import Path

import cv2


VIDEOS_ROOT = Path("/data/users/wjq/datasets/shanghaitech/training/videos")
FRAMES_ROOT = Path("/data/users/wjq/datasets/shanghaitech/training/frames")


def extract_all_frames(video_path: Path, output_dir: Path) -> int:
    """抽取单个视频的所有帧，返回抽取帧数。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Skip] 无法打开视频: {video_path}")
        return 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_name = f"{frame_idx:06d}.jpg"
        frame_path = output_dir / frame_name
        cv2.imwrite(str(frame_path), frame)
        frame_idx += 1

    cap.release()
    return frame_idx


def main() -> None:
    if not VIDEOS_ROOT.exists():
        raise FileNotFoundError(f"视频目录不存在: {VIDEOS_ROOT}")

    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)

    avi_files = sorted(VIDEOS_ROOT.glob("*.avi"))
    if not avi_files:
        print(f"未找到 .avi 视频: {VIDEOS_ROOT}")
        return

    print(f"发现视频数量: {len(avi_files)}")
    total_frames = 0

    for video_path in avi_files:
        video_name = video_path.stem
        out_dir = FRAMES_ROOT / video_name

        count = extract_all_frames(video_path, out_dir)
        total_frames += count
        print(f"[Done] {video_name}: {count} 帧 -> {out_dir}")

    print(f"全部完成，总帧数: {total_frames}")


if __name__ == "__main__":
    main()
