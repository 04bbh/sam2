import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove the final Chinese-period-delimited sentence from UCF3 "
            "video_context fields."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=Path("video_desc/ucf3"),
        type=Path,
        help="Directory containing source JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("video_desc/ucf3_nopredict"),
        type=Path,
        help="Directory to write processed JSON files.",
    )
    return parser.parse_args()


def remove_last_sentence(text: str) -> str:
    sentences = [sentence for sentence in text.split("。") if sentence]
    if len(sentences) < 2:
        raise ValueError("video_context must contain at least two sentences")

    kept_sentences = sentences[:-1]
    return "。".join(kept_sentences) + "。" if kept_sentences else ""


def update_raw_output(raw_output: object, video_context: str) -> object:
    if not isinstance(raw_output, str):
        return raw_output

    try:
        raw_data = json.loads(raw_output)
    except json.JSONDecodeError:
        return raw_output

    if isinstance(raw_data, dict) and "video_context" in raw_data:
        raw_data["video_context"] = video_context
        return json.dumps(raw_data, ensure_ascii=False, indent=4)

    return raw_output


def process_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    try:
        video_context = data["video_desc"]["video_context"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{input_path} missing video_desc.video_context") from error

    if not isinstance(video_context, str):
        raise ValueError(f"{input_path} video_desc.video_context must be a string")

    new_video_context = remove_last_sentence(video_context)
    data["video_desc"]["video_context"] = new_video_context

    if "raw_output" in data:
        data["raw_output"] = update_raw_output(data["raw_output"], new_video_context)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no JSON files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in json_paths:
        process_file(input_path, output_dir / input_path.name)

    print(f"[Done] processed {len(json_paths)} files: {input_dir} -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
