#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, List


def collect_confidences(obj: Any, out: List[float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "confidence" and isinstance(v, (int, float)):
                out.append(float(v))
            else:
                collect_confidences(v, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_confidences(item, out)


def calc_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算目录下每个 JSON 文件的 confidence 均值，并写入 txt。"
    )
    parser.add_argument(
        "--input-dir",
        default="output_json_detector/detector_stage2_json_chaserunmove_1",
        help="待处理 JSON 目录",
    )
    parser.add_argument(
        "--output-file",
        default="./confidence_mean_2.txt",
        help="输出 txt 文件路径",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不是目录: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    lines = []

    for json_file in json_files:
        with json_file.open("r", encoding=args.encoding) as f:
            data = json.load(f)

        values: List[float] = []
        collect_confidences(data, values)
        mean_value = calc_mean(values)
        lines.append(f"{json_file.stem},{mean_value:.3f}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding=args.encoding) as f:
        f.write("\n".join(lines))

    print(f"已输出: {output_file}")
    print(f"处理 JSON 数量: {len(json_files)}")


if __name__ == "__main__":
    main()
