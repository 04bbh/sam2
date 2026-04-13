import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# =========================
# 本地模型配置
# =========================
# 1) 如果你已经把模型下载到本地目录，请把 MODEL_PATH 改成该目录
# 2) 如果可联网，也可以直接填 HuggingFace/ModelScope 的模型名
MODEL_PATH = os.getenv("QWEN3_VL_MODEL_PATH", "Qwen3-VL-8B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

_model = None
_processor = None


def load_local_qwen_vl(model_path: str = MODEL_PATH):
    """
    加载本地部署的 Qwen3-VL 模型（仅首次加载）。
    """
    global _model, _processor
    if _model is not None and _processor is not None:
        return _model, _processor

    print(f"正在加载本地模型: {model_path}")
    _processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        trust_remote_code=True,
        device_map="auto" if DEVICE == "cuda" else None,
    )

    if DEVICE != "cuda":
        _model = _model.to(DEVICE)

    _model.eval()
    print(f"模型加载完成，推理设备: {DEVICE}")
    return _model, _processor


def call_qwen_vl_grounding(image_path, prompt, max_new_tokens=256):
    """
    使用本地部署的 Qwen3-VL 进行视觉定位推理。
    """
    model, processor = load_local_qwen_vl()

    local_path = image_path.replace("file://", "")
    if not os.path.exists(local_path):
        print(f"图片不存在: {local_path}")
        return None

    image = Image.open(local_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    if DEVICE == "cuda":
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 去掉输入 prompt 对应的 token，只保留模型新生成内容
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def parse_and_visualize(image_path, api_response, output_path="result.jpg"):
    """
    解析 Qwen-VL 返回的特殊格式坐标，并画在图上
    Qwen-VL 返回格式通常包含: <box_2d>[y1, x1, y2, x2]</box_2d>
    坐标是归一化到 [0, 1000] 的整数
    """
    if not api_response:
        return

    if isinstance(api_response, list):
        text_parts = []
        for item in api_response:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        api_response = "".join(text_parts) if text_parts else str(api_response)
    elif not isinstance(api_response, str):
        api_response = str(api_response)

    if image_path.startswith("http"):
        print("可视化仅支持本地文件路径用于绘图。请手动检查文本输出。")
        print(f"Model Output: {api_response}")
        return

    local_path = image_path.replace("file://", "")
    img = cv2.imread(local_path)
    if img is None:
        print("无法读取图片")
        return

    h_img, w_img = img.shape[:2]

    patterns = [
        r"<box_2d>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box_2d>",
        r"<box_2d>\[(\d+),(\d+),(\d+),(\d+)\]</box_2d>",
        r"box_2d>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box_2d>",
        r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]",
        r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
        r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)",
    ]

    matches = []
    for pattern in patterns:
        matches = re.findall(pattern, api_response)
        if matches:
            print(f"使用模式匹配成功: {pattern}")
            break

    print(f"\n--- 模型原始输出 ---\n{api_response}")
    print(f"\n--- 解析到的目标数量: {len(matches)} ---")

    if len(matches) == 0:
        print("⚠️  警告：未找到任何坐标信息！")
        print("可能的原因：")
        print("1. 返回格式不是预期的 <box_2d> 格式")
        print("2. Prompt 没有引导模型输出坐标")
        print("3. 图像中没有检测到目标")
        cv2.imwrite(output_path, img)
        return

    for i, box in enumerate(matches):
        try:
            x1_n, y1_n, x2_n, y2_n = map(int, box)

            x1 = int(x1_n / 1000 * w_img)
            y1 = int(y1_n / 1000 * h_img)
            x2 = int(x2_n / 1000 * w_img)
            y2 = int(y2_n / 1000 * h_img)

            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y2 = max(0, min(y2, h_img - 1))

            if x2 <= x1 or y2 <= y1:
                print(f"⚠️  警告：目标 {i+1} 坐标无效，跳过")
                continue

            print(
                f"目标 {i+1}: 归一化坐标 [{x1_n}, {y1_n}, {x2_n}, {y2_n}] -> "
                f"像素坐标 (x1={x1}, y1={y1}, x2={x2}, y2={y2})"
            )

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"Obj {i+1}",
                (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        except Exception as e:
            print(f"⚠️  处理目标 {i+1} 时出错: {e}")
            print(f"   原始坐标数据: {box}")
            continue

    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存至: {output_path}")


if __name__ == "__main__":
    image_file = "/data/users/wjq/codes/sam2-main/sam2-main/test_videos/01_0014/178.jpg"
    img_path_input = f"file://{os.path.abspath(image_file)}"

    prompt_explicit = (
        "请给出图像中“骑自行车的人（含自行车）”的空间坐标，"
        "请以 <box_2d>[x_min, ymin, x_max, y_max]</box_2d> 格式"
        "（数值范围 0-1000）返回 bounding box。"
    )
    #爆炸（含爆炸后的烟雾）

    result_text = call_qwen_vl_grounding(img_path_input, prompt_explicit)
    parse_and_visualize(image_file, result_text)