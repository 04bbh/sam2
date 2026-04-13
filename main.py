import argparse
import os
from typing import List

import yaml
import torch
from src.detector import QwenVLDetector
from src.pipeline import AsyncPipelineRunner
from src.selector import FrameSelector
from src.segmenter_and_tracker import SegmenterAndTracker
from utils.visualization import save_tracking_results

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_video_dirs(videos_root: str) -> List[str]:
    subdirs = []
    for name in sorted(os.listdir(videos_root)):
        full = os.path.join(videos_root, name)
        if os.path.isdir(full):
            subdirs.append(full)
    return subdirs

def list_frames(video_dir: str):
    names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    names.sort()
    return [os.path.join(video_dir, n) for n in names]

def run_single_video(segmenter_and_tracker: SegmenterAndTracker, pipeline: AsyncPipelineRunner, video_dir: str, cfg: dict) -> None:
    frame_paths = list_frames(video_dir)
    if not frame_paths:
        print(f"[Skip] 无帧图像: {video_dir}")
        return

    pipe_result = pipeline.run(frame_paths=frame_paths, batch_size=cfg["pipeline"]["batch_size"])

    selector = FrameSelector(alpha=cfg["selector"]["alpha"], beta=cfg["selector"]["beta"])
    best = selector.select_best(pipe_result.segmentations)

    print(
        f"[Best] video={os.path.basename(video_dir)} frame_idx={best.frame_idx} "
        f"score={best.score:.4f} (qwen_conf={best.confidence:.4f}, sam_iou={best.iou_score:.4f})"
    )

    video_segments = segmenter_and_tracker.track_from_mask(
        video_dir=video_dir,
        ann_frame_idx=best.frame_idx,
        ann_obj_id=int(cfg["data"]["ann_obj_id"]),
        mask=best.mask,
    )

    out_dir = os.path.join(cfg["data"]["output_root"], os.path.basename(video_dir))
    save_tracking_results(video_dir=video_dir, video_segments=video_segments, out_dir=out_dir)
    print(f"[Done] 保存结果: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", type=str)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    
    # 加载qwen3-vl模型
    detector = QwenVLDetector(
        model_path=cfg["qwen"]["model_path"],
        target_desc=cfg["qwen"]["target_desc"],
        max_new_tokens=cfg["qwen"]["max_new_tokens"],
        use_quantization=False,
        use_flashattn=True,
        device=device,
        batch_size=cfg["pipeline"]["batch_size"],
    )
    
    # 加载sam2模型
    segmenter_and_tracker = SegmenterAndTracker(
        model_cfg=cfg["sam2"]["model_cfg"],
        checkpoint=cfg["sam2"]["checkpoint"],
        multimask_output=cfg["sam2"]["multimask_output"],
        device=device,
    )

    pipeline = AsyncPipelineRunner(
        detector=detector,
        segmenter_and_tracker=segmenter_and_tracker,
        queue_size=cfg["pipeline"]["queue_size"],
    )

    videos_root = cfg["data"]["videos_root"]
    video_dirs = discover_video_dirs(videos_root)
    if not video_dirs:
        print(f"未发现视频目录: {videos_root}")
        return

    for video_dir in video_dirs:
        try:
            run_single_video(segmenter_and_tracker, pipeline, video_dir, cfg)
        except Exception as e:
            print(f"[Error] {video_dir}: {e}")


if __name__ == "__main__":
    main()
