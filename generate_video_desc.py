import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

DEFAULT_VIDEO_ROOT = "/data/users/lzh/datasets/PreVAD/other_datasets/ucf_videos"
DEFAULT_MODEL_PATH = "Qwen3-VL-8B-Instruct"
DEFAULT_OUTPUT_DIR = "video_desc/ucf3"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate objective video-level descriptions with local Qwen3-VL."
    )
    parser.add_argument(
        "--video_root",
        default=DEFAULT_VIDEO_ROOT,
        type=str,
        help="Root directory for batch video processing.",
    )
    parser.add_argument(
        "--video_path",
        default=None,
        type=str,
        help="Optional single video path. If set, this takes precedence over --video_root.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        type=str,
        help="Directory where description JSON files are saved.",
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        type=str,
        help="Local Qwen3-VL model path.",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=512,
        type=int,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--fps",
        default=2,
        type=float,
        help="Video sampling FPS passed to Qwen3-VL processor.",
    )
    parser.add_argument(
        "--max_pixels",
        default=448 * 448,
        type=int,
        help="Maximum pixels per sampled video frame. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default="2,3",
        type=str,
        help="Optional CUDA_VISIBLE_DEVICES value set before importing torch.",
    )
    parser.add_argument(
        "--skip_existing",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Skip videos whose output JSON already exists.",
    )
    return parser.parse_args()


def empty_video_desc() -> dict[str, Any]:
    return {
        "video_context": "",
    }


def dependency_error_message(exc: BaseException) -> str:
    missing = getattr(exc, "name", None)
    if missing:
        return (
            f"Missing dependency: {missing}. Please install Qwen3-VL video dependencies, "
            "for example: pip install -U transformers torch torchvision torchcodec "
            "librosa accelerate regex qwen-vl-utils"
        )
    return (
        "Failed to import Qwen3-VL dependencies. Please make sure transformers, torch, "
        "torchvision, torchcodec, librosa, accelerate, regex, and qwen-vl-utils are installed. "
        f"Original error: {exc}"
    )


def import_qwen_dependencies():
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from qwen_vl_utils import process_vision_info
    except ModuleNotFoundError as exc:
        raise RuntimeError(dependency_error_message(exc)) from exc
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import AutoModelForImageTextToText from transformers. "
            "Qwen3-VL inference requires a recent transformers build. "
            f"Original error: {exc}"
        ) from exc
    return torch, AutoProcessor, AutoModelForImageTextToText, process_vision_info


def find_video_files(video_root: str) -> list[Path]:
    root = Path(video_root)
    if not root.exists():
        raise FileNotFoundError(f"Video root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Video root is not a directory: {root}")

    videos = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos, key=lambda path: str(path))


def resolve_video_paths(args: argparse.Namespace) -> list[Path]:
    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
        if not video_path.is_file():
            raise ValueError(f"Video path is not a file: {video_path}")
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported video extension: {video_path.suffix}. "
                f"Supported extensions: {sorted(VIDEO_EXTENSIONS)}"
            )
        return [video_path]

    videos = find_video_files(args.video_root)
    if not videos:
        raise FileNotFoundError(
            f"No video files found under {args.video_root}. "
            f"Supported extensions: {sorted(VIDEO_EXTENSIONS)}"
        )
    return videos


def build_prompt() -> str:
    return """
    # Task
    请根据输入视频，生成一段客观、简洁的视频级时序描述，并推测是否存在Detection Categories中定义的事件类别。

    # Detection Categories（不同事件类别用“，”分隔）
    "虐待动物，逮捕，着火，纵火，打斗，车祸，爆炸，抢劫，枪击，蓄意破坏，偷窃"

    # Rules
    - 时序描述应概括视频整体过程，主要包含画面中可见场景、主要对象、动作变化和交互过程，尽量描述清楚动作细节。
    - 时序描述不要包含静态背景（例如画面颜色、视角固定、停放车辆）。
    - 如果画面模糊、遮挡、远景、低光或镜头抖动，并影响动作判断，要说明不确定性。
    - 输出 4 到 5 句话，最后一句话一定是推测的事件类别。若视频中不存在Detection Categories中定义的事件类别，推测结果为“当前视频大概率不包含定义的事件类别”。

   
    # Output Format
    严格输出 JSON，不要输出 Markdown，不要输出解释文字。
    JSON 只能包含以下字段：
    {
    "video_context": string
    }

    # Output Example

    {
    "video_context": "一个男人走向柜台区域并爬上柜台，随后多次抬手靠近柜台并取走物品，最后跳下柜台，消失在画面中。当前视频可能存在“偷窃”事件。"
    }

    {
    "video_context": "多名顾客在超市收银台前排队结账，工作人员在各自岗位上扫描商品并打包。一名顾客完成结账后提着购物袋离开。当前视频大概率不包含定义的事件类别。"
    }
    """.strip()
# - 不要判断视频属于哪一种事件类别。
# - 不要使用“抢劫、偷窃、逮捕、打斗、纵火、枪击、爆炸、车祸”等事件类别词作为结论。
# - 不推测人物身份、意图、违法性质或不可见因果关系。

# # Rules
# - 只描述画面中可见的场景、主要对象、动作变化、空间关系和交互过程。
# - 描述应概括视频整体过程，不要逐帧罗列，也不要写成动作类别归纳。
# - 尽量描述清楚动作细节
# - 不要推测视频的因果关系，仅给出客观描述
# - 不要描述静态背景或未参与动作的物体，例如画面颜色、视角固定、地面材质、停放车辆、墙面、门窗等。
# - 不要描述人物的外观、穿着。
# - 如果画面模糊、遮挡、远景、低光或镜头抖动，并影响动作判断，要说明不确定性。
# - 输出 2 到 4 句话。

def load_model(model_path: str):
    torch, AutoProcessor, AutoModelForImageTextToText, process_vision_info = import_qwen_dependencies()
    print(f"[Info] Loading Qwen3-VL model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype="auto",
            **model_kwargs,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype="auto",
            **model_kwargs,
        )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return torch, processor, model, process_vision_info


def decode_response(processor: Any, response: str) -> str:
    parse_response = getattr(processor, "parse_response", None)
    if parse_response is None:
        return response.strip()

    try:
        parsed = parse_response(response)
    except Exception:
        return response.strip()

    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        for key in ("content", "text", "answer", "response"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()
        return json.dumps(parsed, ensure_ascii=False)
    return str(parsed).strip()


def generate_description(
    video_path: Path,
    processor: Any,
    model: Any,
    process_vision_info: Any,
    max_new_tokens: int,
    fps: float,
    max_pixels: int | None,
) -> str:
    video_info = {
        "type": "video",
        "video": f"file://{video_path.resolve()}",
        "fps": fps,
    }
    if max_pixels is not None and max_pixels > 0:
        video_info["max_pixels"] = max_pixels

    messages = [
        {
            "role": "user",
            "content": [
                video_info,
                {"type": "text", "text": build_prompt()},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    processor_kwargs = {
        "text": [text],
        "images": image_inputs,
        "videos": video_inputs,
        "padding": True,
        "return_tensors": "pt",
        "video_metadata": video_metadatas,
    }
    try:
        inputs = processor(**processor_kwargs, **video_kwargs)
    except TypeError:
        processor_kwargs.pop("video_metadata", None)
        inputs = processor(**processor_kwargs, **video_kwargs)

    device = next(model.parameters()).device
    inputs = inputs.to(device)
    input_len = inputs["input_ids"].shape[-1]

    with __import__("torch").inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    if hasattr(processor, "batch_decode"):
        response = processor.batch_decode(
            [outputs[0][input_len:]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    else:
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return decode_response(processor, response)


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)

    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def normalize_video_desc(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return empty_video_desc()

    desc = empty_video_desc()
    value = data.get("video_context", "")
    if isinstance(value, str):
        desc["video_context"] = value.strip()
        return desc

    # Backward-compatible fallback for older prompt outputs.
    parts = []
    for key in ("scene", "motion_summary", "interaction_summary"):
        item = data.get(key, "")
        if isinstance(item, str) and item.strip():
            parts.append(item.strip())
    desc["video_context"] = " ".join(parts)

    return desc


def output_path_for(video_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{video_path.stem}.json"


def save_result(
    video_path: Path,
    model_path: str,
    output_path: Path,
    raw_output: str,
) -> None:
    parsed = extract_json_object(raw_output)
    result = {
        "video_path": str(video_path),
        "video_name": video_path.name,
        "model_path": model_path,
        "video_desc": normalize_video_desc(parsed),
        "raw_output": raw_output,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    try:
        videos = resolve_video_paths(args)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    pending = []
    for video_path in videos:
        out_path = output_path_for(video_path, output_dir)
        if args.skip_existing and out_path.exists():
            print(f"[Skip] Existing output: {out_path}")
            continue
        pending.append((video_path, out_path))

    if not pending:
        print("[Info] No videos to process.")
        return 0

    try:
        _, processor, model, process_vision_info = load_model(args.model_path)
    except Exception as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        return 1

    for index, (video_path, out_path) in enumerate(pending, start=1):
        print(f"[Info] ({index}/{len(pending)}) Processing: {video_path}")
        try:
            raw_output = generate_description(
                video_path=video_path,
                processor=processor,
                model=model,
                process_vision_info=process_vision_info,
                max_new_tokens=args.max_new_tokens,
                fps=args.fps,
                max_pixels=args.max_pixels,
            )
            save_result(
                video_path=video_path,
                model_path=args.model_path,
                output_path=out_path,
                raw_output=raw_output,
            )
            print(f"[Done] Saved: {out_path}")
        except Exception as exc:
            print(f"[Error] Failed to process {video_path}: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
