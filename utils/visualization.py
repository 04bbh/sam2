import os
from typing import Any, Dict, Sequence

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFont = None


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


def _is_empty_box(box_xyxy: np.ndarray) -> bool:
    return np.allclose(box_xyxy, np.zeros(4, dtype=np.float32))


def _load_label_font(size: int = 18):
    if ImageFont is None:
        return None

    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _draw_label(image_bgr: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> np.ndarray:
    if Image is None or ImageDraw is None:
        label_y = max(18, y - 6)
        cv2.putText(
            image_bgr,
            text,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        return image_bgr

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(image_pil)
    font = _load_label_font()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    label_y = max(0, y - text_h - 6)
    bg_box = [x, label_y, x + text_w + 8, label_y + text_h + 6]
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))
    draw.rectangle(bg_box, fill=rgb_color)
    draw.text((x + 4, label_y + 3), text, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def save_detection_results(detections: Sequence[Any], out_dir: str) -> None:
    if not detections:
        return

    os.makedirs(out_dir, exist_ok=True)
    grouped: Dict[str, list[Any]] = {}
    for det in detections:
        grouped.setdefault(det.frame_path, []).append(det)

    palette = [
        (30, 144, 255),
        (46, 204, 113),
        (255, 99, 71),
        (255, 215, 0),
        (186, 85, 211),
    ]

    for frame_path, frame_detections in grouped.items():
        img = cv2.imread(frame_path)
        if img is None:
            continue

        valid_detections = [
            det for det in frame_detections if not _is_empty_box(np.asarray(det.box_xyxy, dtype=np.float32))
        ]
        if not valid_detections:
            img = _draw_label(img, "no detection", 8, 28, (80, 80, 80))
        else:
            h, w = img.shape[:2]
            for i, det in enumerate(valid_detections):
                color = palette[i % len(palette)]
                x1, y1, x2, y2 = np.asarray(det.box_xyxy, dtype=np.float32).astype(int).tolist()
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                name = det.category if det.category else "det"
                img = _draw_label(img, f"{name} {det.confidence:.2f}", x1, y1, color)

        save_path = os.path.join(out_dir, os.path.basename(frame_path))
        cv2.imwrite(save_path, img)
