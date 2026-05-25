#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT_DIR = (
    "output_detector_json/output_json_detector_xd/detector_stage2_json_1"
)
DEFAULT_OUTPUT_DIR = (
    "output_detector_json/output_json_detector_xd/detector_stage2_json_1_nosmoke"
)
DEFAULT_REMOVE_CATEGORY = "烟雾"


def filter_detections(
    frames: List[Dict[str, Any]], remove_category: str
) -> Tuple[List[Dict[str, Any]], int]:
    removed_count = 0

    for frame in frames:
        detections = frame.get("detections")
        if not isinstance(detections, list):
            continue

        kept_detections = []
        for detection in detections:
            if (
                isinstance(detection, dict)
                and detection.get("category") == remove_category
            ):
                removed_count += 1
                continue
            kept_detections.append(detection)

        frame["detections"] = kept_detections

    return frames, removed_count


def process_file(
    input_file: Path, output_file: Path, remove_category: str, encoding: str
) -> int:
    with input_file.open("r", encoding=encoding) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 顶层结构不是列表")

    filtered_data, removed_count = filter_detections(data, remove_category)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding=encoding) as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return removed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="删除检测 JSON 中指定类别的 detections，并写入新目录。"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="待处理 JSON 目录",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="过滤后 JSON 输出目录",
    )
    parser.add_argument(
        "--remove-category",
        default=DEFAULT_REMOVE_CATEGORY,
        help="需要从 detections 中删除的 category",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    processed_count = 0
    removed_count = 0
    error_count = 0

    for json_file in json_files:
        output_file = output_dir / json_file.name
        try:
            removed_count += process_file(
                json_file, output_file, args.remove_category, args.encoding
            )
            processed_count += 1
        except Exception as exc:
            error_count += 1
            print(f"跳过文件: {json_file}，原因: {exc}")

    print(f"处理 JSON 数量: {processed_count}")
    print(f"删除 {args.remove_category} 检测项数量: {removed_count}")
    print(f"错误 JSON 数量: {error_count}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
