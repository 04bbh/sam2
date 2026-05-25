import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def normalize_score(score, eps=1e-8):
    score = np.asarray(score, dtype=np.float32)
    min_v = float(score.min())
    max_v = float(score.max())

    if max_v - min_v < eps:
        return np.zeros_like(score, dtype=np.float32)

    return (score - min_v) / (max_v - min_v + eps)


def split_frames_to_intervals(frames, max_gap=1):
    """
    Split discontinuous frame indices into multiple temporal intervals.

    Args:
        frames:
            1D array of frame indices.
        max_gap:
            If the gap between two adjacent frames is larger than max_gap,
            a new interval is started.
            max_gap=1 means only strictly continuous frames are merged.
            max_gap=3 means small missing gaps are tolerated.

    Returns:
        intervals:
            List of (start_frame, end_frame).
    """
    frames = np.asarray(frames, dtype=int)
    frames = np.unique(np.sort(frames))

    if len(frames) == 0:
        return []

    intervals = []
    start = frames[0]
    prev = frames[0]

    for f in frames[1:]:
        if f - prev > max_gap:
            intervals.append((int(start), int(prev)))
            start = f
        prev = f

    intervals.append((int(start), int(prev)))
    return intervals


def make_local_lognormal_curve(
    total_frames,
    start_f,
    end_f,
    r=0.2,
    sigma_k=0.55,
    smooth_sigma=3.0,
    pre_margin_ratio=0.05,
    post_margin_ratio=0.20,
):
    """
    Generate a local-coordinate Log-normal curve for one interval.
    """

    score = np.zeros(total_frames, dtype=np.float32)

    start_f = max(0, min(int(start_f), total_frames - 1))
    end_f = max(0, min(int(end_f), total_frames - 1))

    if end_f <= start_f:
        score[start_f] = 1.0
        return score, float(start_f)

    interval_len = end_f - start_f

    pre_margin = int(pre_margin_ratio * interval_len)
    post_margin = int(post_margin_ratio * interval_len)

    left = max(0, start_f - pre_margin)
    right = min(total_frames - 1, end_f + post_margin)

    region_len = right - left + 1
    x_local = np.arange(1, region_len + 1, dtype=np.float32)

    # Peak position in global frame index.
    peak_frame = start_f + r * interval_len

    # Convert global peak to local coordinate.
    peak_local = peak_frame - left + 1
    peak_local = max(peak_local, 1.0)

    # For Log-normal distribution:
    # mode = exp(mu - sigma^2)
    # Therefore, mu = log(mode) + sigma^2
    mu = np.log(peak_local) + sigma_k ** 2

    local_curve = (1.0 / (x_local * sigma_k)) * np.exp(
        -((np.log(x_local) - mu) ** 2) / (2.0 * sigma_k ** 2)
    )

    local_curve = normalize_score(local_curve)

    if smooth_sigma is not None and smooth_sigma > 0:
        local_curve = gaussian_filter1d(local_curve, sigma=smooth_sigma)
        local_curve = normalize_score(local_curve)

    score[left:right + 1] = local_curve

    return score, float(peak_frame)


def trajectory_to_multi_lognormal_ps(
    txt_path,
    total_frames,
    max_gap=1,
    r=0.2,
    sigma_k=0.55,
    smooth_sigma=3.0,
    pre_margin_ratio=0.05,
    post_margin_ratio=0.20,
    aggregation="max",
):
    """
    Convert a trajectory txt file with possibly multiple discontinuous temporal
    intervals to frame-level anomaly scores.

    Input txt format:
        frame_id, x1, y1, x2, y2, confidence

    Note:
        The confidence column is ignored.

    Args:
        txt_path:
            Path to trajectory txt.

        total_frames:
            Total number of frames in the video.

        max_gap:
            Gap threshold for splitting intervals.
            If adjacent frame gap > max_gap, split into a new interval.

        aggregation:
            How to merge multiple interval curves.
            "max": recommended, avoids over-amplification.
            "sum": sum all curves and normalize.
            "prob": probabilistic union, 1 - prod(1 - score_i).

    Returns:
        final_score:
            Frame-level anomaly score, shape [total_frames].

        intervals:
            List of detected temporal intervals.

        peak_frames:
            List of peak frames for each interval.
    """

    data = pd.read_csv(txt_path, header=None)

    if data.shape[1] < 5:
        raise ValueError(
            "The input txt should contain at least 5 columns: "
            "frame_id, x1, y1, x2, y2."
        )

    frames = data.iloc[:, 0].values.astype(int)
    frames = frames[(frames >= 0) & (frames < total_frames)]

    intervals = split_frames_to_intervals(frames, max_gap=max_gap)

    if len(intervals) == 0:
        return np.zeros(total_frames, dtype=np.float32), [], []

    curves = []
    peak_frames = []

    for start_f, end_f in intervals:
        curve, peak = make_local_lognormal_curve(
            total_frames=total_frames,
            start_f=start_f,
            end_f=end_f,
            r=r,
            sigma_k=sigma_k,
            smooth_sigma=smooth_sigma,
            pre_margin_ratio=pre_margin_ratio,
            post_margin_ratio=post_margin_ratio,
        )
        curves.append(curve)
        peak_frames.append(peak)

    curves = np.stack(curves, axis=0)

    if aggregation == "max":
        final_score = np.max(curves, axis=0)

    elif aggregation == "sum":
        final_score = np.sum(curves, axis=0)

    elif aggregation == "prob":
        final_score = 1.0 - np.prod(1.0 - np.clip(curves, 0, 1), axis=0)

    else:
        raise ValueError("aggregation should be one of: 'max', 'sum', 'prob'.")

    final_score = normalize_score(final_score)

    return final_score.astype(np.float32), intervals, peak_frames


def visualize_multi_score(score, intervals, peak_frames, save_path=None):
    plt.figure(figsize=(12, 4))
    plt.plot(score, linewidth=2, label="Multi-interval Log-normal PS score")

    for idx, (start_f, end_f) in enumerate(intervals):
        plt.axvline(
            start_f,
            linestyle="--",
            linewidth=1.2,
            label=f"Start {idx + 1}: {start_f}",
        )
        plt.axvline(
            end_f,
            linestyle="--",
            linewidth=1.2,
            label=f"End {idx + 1}: {end_f}",
        )

    for idx, peak in enumerate(peak_frames):
        plt.axvline(
            peak,
            linestyle=":",
            linewidth=1.8,
            label=f"Peak {idx + 1}: {peak:.1f}",
        )

    plt.xlabel("Frame index")
    plt.ylabel("Anomaly score")
    plt.title("Frame-level Anomaly Score by Multi-interval Log-normal PS")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Visualization saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    txt_path = "/data/users/wjq/codes/sam2-main/output_tracks/output_tracks_ucf/improved_3_0.9_bs8/Arson016_x264.txt"
    total_frames = 1795

    score, intervals, peak_frames = trajectory_to_multi_lognormal_ps(
        txt_path=txt_path,
        total_frames=total_frames,

        # 如果中间断 1 帧就算断开，用 max_gap=1。
        # 如果 SAM2 偶尔漏几帧，可以设为 3 或 5。
        max_gap=1,

        # Log-normal 参数
        r=0.2,
        sigma_k=0.55,
        smooth_sigma=3.0,

        # 开始前少量提前响应，结束后允许更长拖尾
        pre_margin_ratio=0.05,
        post_margin_ratio=0.20,

        # 多段曲线聚合方式
        aggregation="max",
    )

    print("Intervals:", intervals)
    print("Peak frames:", peak_frames)
    print("Score shape:", score.shape)
    print("Score min:", score.min())
    print("Score max:", score.max())

    np.save("Arson016_x264_multi_lognormal_score.npy", score)
    np.savetxt("Arson016_x264_multi_lognormal_score.txt", score, fmt="%.6f")

    visualize_multi_score(
        score=score,
        intervals=intervals,
        peak_frames=peak_frames,
        save_path="Arson016_x264_multi_lognormal_score.png",
    )