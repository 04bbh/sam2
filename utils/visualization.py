import os
from typing import Dict

import cv2
import numpy as np


def overlay_mask_on_image(image_bgr: np.ndarray, mask: np.ndarray, color=(30, 144, 255), alpha: float = 0.5):
    mask = mask.astype(bool)
    h, w = image_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = np.zeros_like(image_bgr, dtype=np.uint8)
    overlay[mask] = np.array(color, dtype=np.uint8)
    blended = cv2.addWeighted(image_bgr, 1.0, overlay, alpha, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)
    return blended


def save_tracking_results(video_dir: str, video_segments: Dict[int, Dict[int, np.ndarray]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    frame_names.sort()

    for i, frame_name in enumerate(frame_names):
        frame_path = os.path.join(video_dir, frame_name)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        if i in video_segments:
            for _, mask in video_segments[i].items():
                img = overlay_mask_on_image(img, mask)

        save_path = os.path.join(out_dir, frame_name)
        cv2.imwrite(save_path, img)
