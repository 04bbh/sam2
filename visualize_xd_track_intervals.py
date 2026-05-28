import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TRACK_DIR = Path(
    "/data/users/wjq/codes/sam2-main/output_tracks/output_tracks_xd/improved_1"
)
DEFAULT_FRAMES_DIR = Path("/data/users/wjq/datasets/XD/frames")
DEFAULT_OUTPUT_DIR = Path(
    "/data/users/wjq/codes/sam2-main/output_vis/output_loc_xd/improved_1_visualization"
)
FRAME_INDEX_SCALE = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize XD track frame intervals as red timeline spans."
    )
    parser.add_argument(
        "--track-dir",
        default=DEFAULT_TRACK_DIR,
        type=Path,
        help="Directory containing track txt files.",
    )
    parser.add_argument(
        "--frames-dir",
        default=DEFAULT_FRAMES_DIR,
        type=Path,
        help="Directory containing per-video frame folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory to write interval plot png files.",
    )
    return parser.parse_args()


def count_frames(frames_path: Path) -> int:
    if not frames_path.is_dir():
        raise FileNotFoundError(f"missing frames directory: {frames_path}")

    frame_count = sum(1 for path in frames_path.iterdir() if path.is_file())
    if frame_count <= 0:
        raise ValueError(f"frames directory is empty: {frames_path}")
    return frame_count


def read_frame_indices(track_path: Path) -> list[int]:
    frame_indices: list[int] = []
    with track_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            first_column = line.split(",", maxsplit=1)[0]
            try:
                frame_indices.append(int(first_column))
            except ValueError as error:
                raise ValueError(
                    f"{track_path} line {line_number} has invalid frame index: "
                    f"{first_column!r}"
                ) from error

    if not frame_indices:
        raise ValueError(f"track file is empty: {track_path}")
    return sorted(set(frame_indices))


def merge_intervals(frame_indices: list[int]) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    start = previous = frame_indices[0]

    for frame_index in frame_indices[1:]:
        if frame_index == previous + 1:
            previous = frame_index
            continue

        intervals.append((start, previous))
        start = previous = frame_index

    intervals.append((start, previous))
    return intervals


def plot_intervals(
    title: str,
    frame_count: int,
    intervals: list[tuple[int, int]],
    output_path: Path,
) -> None:
    width, height = 1800, 500
    left, right, top, bottom = 90, 30, 50, 70
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    max_x_value = max((frame_count - 1) / FRAME_INDEX_SCALE, 1.0)

    image = np.full((height, width, 3), 255, dtype=np.uint8)

    def x_to_px(x_value: float) -> int:
        return int(round(plot_left + (x_value / max_x_value) * plot_width))

    def y_to_px(value: float) -> int:
        return int(round(plot_bottom - value * plot_height))

    grid_color = (220, 220, 220)
    axis_color = (0, 0, 0)
    text_color = (0, 0, 0)
    red = (0, 0, 255)
    fill_color = (209, 209, 255)

    for tick in np.linspace(0, max_x_value, 8):
        x = x_to_px(float(tick))
        cv2.line(image, (x, plot_top), (x, plot_bottom), grid_color, 1)
        cv2.putText(
            image,
            f"{tick:.1f}",
            (x - 15, plot_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )

    for tick in np.linspace(0, 1, 6):
        y = y_to_px(float(tick))
        cv2.line(image, (plot_left, y), (plot_right, y), grid_color, 1)
        cv2.putText(
            image,
            f"{tick:.1f}",
            (plot_left - 55, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )

    for start, end in intervals:
        x1 = max(plot_left, min(plot_right, x_to_px(start / FRAME_INDEX_SCALE)))
        x2 = max(plot_left, min(plot_right, x_to_px((end + 1) / FRAME_INDEX_SCALE)))
        if x2 <= x1:
            x2 = min(plot_right, x1 + 1)
        cv2.rectangle(image, (x1, plot_top), (x2, plot_bottom), fill_color, -1)
        cv2.rectangle(image, (x1, plot_top), (x2, plot_bottom), red, 1)

    cv2.rectangle(image, (plot_left, plot_top), (plot_right, plot_bottom), axis_color, 1)

    title_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(
        image,
        title,
        ((width - title_size[0]) // 2, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Frame index / 8",
        ((width - 160) // 2, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Score",
        (15, plot_top + plot_height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        1,
        cv2.LINE_AA,
    )

    legend_x, legend_y = width - 250, 65
    cv2.rectangle(image, (legend_x, legend_y), (width - 45, legend_y + 45), (245, 245, 245), -1)
    cv2.rectangle(image, (legend_x, legend_y), (width - 45, legend_y + 45), (210, 210, 210), 1)
    cv2.rectangle(image, (legend_x + 15, legend_y + 15), (legend_x + 45, legend_y + 30), fill_color, -1)
    cv2.rectangle(image, (legend_x + 15, legend_y + 15), (legend_x + 45, legend_y + 30), red, 1)
    cv2.putText(
        image,
        "track interval",
        (legend_x + 55, legend_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        1,
        cv2.LINE_AA,
    )

    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"failed to write image: {output_path}")


def process_track_file(track_path: Path, frames_dir: Path, output_dir: Path) -> None:
    frame_count = count_frames(frames_dir / track_path.stem)
    frame_indices = read_frame_indices(track_path)
    intervals = merge_intervals(frame_indices)
    plot_intervals(
        title=track_path.stem,
        frame_count=frame_count,
        intervals=intervals,
        output_path=output_dir / f"{track_path.stem}.png",
    )


def main() -> int:
    args = parse_args()
    track_dir = args.track_dir
    frames_dir = args.frames_dir
    output_dir = args.output_dir

    if not track_dir.is_dir():
        raise FileNotFoundError(f"track directory does not exist: {track_dir}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory does not exist: {frames_dir}")

    track_paths = sorted(track_dir.glob("*.txt"))
    if not track_paths:
        raise FileNotFoundError(f"no txt files found in: {track_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, track_path in enumerate(track_paths, start=1):
        process_track_file(track_path, frames_dir, output_dir)
        if index % 50 == 0 or index == len(track_paths):
            print(f"[Progress] {index}/{len(track_paths)}")

    print(f"[Done] generated {len(track_paths)} plots: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
