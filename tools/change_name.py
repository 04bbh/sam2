import re
from pathlib import Path


FRAMES_ROOT = Path("/data/users/wjq/datasets/shanghaitech/training/frames")
NAME_PATTERN = re.compile(r"^(\d{6})\.jpg$", re.IGNORECASE)


def rename_in_dir(video_dir: Path) -> int:
    """将单个视频目录中的 000000.jpg 重命名为 000.jpg，返回重命名数量。"""
    renamed = 0

    # 先收集并排序，避免遍历时文件名变化影响结果
    files = sorted(p for p in video_dir.iterdir() if p.is_file())

    for src in files:
        m = NAME_PATTERN.match(src.name)
        if not m:
            continue

        six_digits = m.group(1)
        # 000000 -> 000, 000123 -> 123, 001234 -> 1234
        target_stem = str(int(six_digits)).zfill(3)
        dst = src.with_name(f"{target_stem}.jpg")

        if dst == src:
            continue

        if dst.exists():
            print(f"[Skip] 目标文件已存在: {dst}")
            continue

        src.rename(dst)
        renamed += 1

    return renamed


def main() -> None:
    if not FRAMES_ROOT.exists():
        raise FileNotFoundError(f"目录不存在: {FRAMES_ROOT}")

    video_dirs = sorted(p for p in FRAMES_ROOT.iterdir() if p.is_dir())
    if not video_dirs:
        print(f"未找到视频帧子目录: {FRAMES_ROOT}")
        return

    total = 0
    for video_dir in video_dirs:
        cnt = rename_in_dir(video_dir)
        total += cnt
        print(f"[Done] {video_dir.name}: 重命名 {cnt} 个文件")

    print(f"全部完成，总重命名数量: {total}")


if __name__ == "__main__":
    main()
