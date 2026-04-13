import threading
from dataclasses import dataclass
from queue import Queue
from typing import List, Sequence

from src.detector import DetectionResult, QwenVLDetector
from src.segmenter_and_tracker import SegmenterAndTracker, SegmentationResult


@dataclass
class PipelineResult:
    detections: List[DetectionResult]
    segmentations: List[SegmentationResult]


class AsyncPipelineRunner:
    """
    Qwen3-VL 与 SAM2 的生产者-消费者流水线：
    - 生产者：按 batch 跑 Qwen，得到 DetectionResult
    - 消费者：拿到 detection 立即跑 SAM2 image predictor
    """

    def __init__(self, detector: QwenVLDetector, segmenter_and_tracker: SegmenterAndTracker, queue_size: int = 32) -> None:
        self.detector = detector
        self.segmenter_and_tracker = segmenter_and_tracker
        self.queue_size = queue_size

    def run(self, frame_paths: Sequence[str], batch_size: int) -> PipelineResult:
        if batch_size <= 0:
            raise ValueError("batch_size 必须 > 0")

        q: Queue = Queue(maxsize=self.queue_size)
        detections: List[DetectionResult] = []
        segmentations: List[SegmentationResult] = []
        lock = threading.Lock()
        stop_token = object()

        def producer() -> None:
            n = len(frame_paths)
            for st in range(0, n, batch_size):
                ed = min(st + batch_size, n)
                batch_paths = frame_paths[st:ed]
                batch_indices = list(range(st, ed))
                det_batch = self.detector.infer_batch(batch_paths, batch_indices)
                print(f"Detector batch {st:04d}-{ed:04d} results: {det_batch}")
                for det in det_batch:
                    q.put(det)
            q.put(stop_token)

        def consumer() -> None:
            while True:
                item = q.get()
                if item is stop_token:
                    q.task_done()
                    break
                det: DetectionResult = item
                seg = self.segmenter_and_tracker.segment(det)
                with lock:
                    detections.append(det)
                    segmentations.append(seg)
                q.task_done()

        t_prod = threading.Thread(target=producer, daemon=True)
        t_cons = threading.Thread(target=consumer, daemon=True)
        t_prod.start()
        t_cons.start()

        t_prod.join()
        q.join()
        t_cons.join()

        detections.sort(key=lambda x: x.frame_idx)
        segmentations.sort(key=lambda x: x.frame_idx)
        return PipelineResult(detections=detections, segmentations=segmentations)
