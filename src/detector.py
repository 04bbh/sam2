import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import BitsAndBytesConfig


@dataclass
class DetectionResult:
    frame_idx: int
    frame_path: str
    box_xyxy: np.ndarray
    confidence: float


class QwenVLDetector:
    """
    Qwen3-VL 检测封装：输出 box + 置信度。

    置信度约束规则通过 prompt 明确给模型：
    - 0.8~1.0：目标清晰可识别
    - 0.4~0.8：存在相似性但有不确定
    - 0.0~0.4：仅能推断存在，无法精准定位
    """

    def __init__(
        self,
        model_path: str,
        target_desc: str,
        max_new_tokens: int = 256,
        use_quantization: bool = False,
        use_flashattn: bool = False,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        self.model_path = model_path
        self.target_desc = target_desc
        self.use_quantization = use_quantization
        self.use_flashattn = use_flashattn
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.batch_size = batch_size

        # 量化配置（可选）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=bnb_config if self.use_quantization else None,
            attn_implementation="flash_attention_2" if self.use_flashattn else "sdpa",
            device_map="auto" if self.device == "cuda" else None,
        )
        # 使用 accelerate 的 device_map="auto" 时，禁止再手动 .to(...)
        # 否则会破坏分片设备放置并触发跨卡 device mismatch。
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        # self.model.eval()

    def _build_prompt(self) -> str:
        
        user_prompt = """
        # Role
        你是一个高精度的视觉目标检测系统。请分析输入的 {batch_size} 张图像，并识别其中包含的所有指定目标。

        # Detection Categories
        请检测以下目标：
        {target_desc}

        # Output Format
        严格输出 JSON 数组，不包含任何 Markdown 代码块标签或解释文字。格式如下：
        [
        {{
            "image_id": 0,
            "box": [xmin, ymin, xmax, ymax],
            "confidence": 0.95
         
        }},
        {{
            "image_id": 1,
            "box": [xmin, ymin, xmax, ymax],
            "confidence": 0.4
        }},
        {{
            "image_id": 2,
            "box": [0, 0, 0, 0],
            "confidence": 0.0
        }}
        ]

        # Strict Rules
        1. 坐标规范：使用 0-1000 的标准化整数，顺序为[x_min,y_min,x_max,y_max]。
        2. 无目标处理：如果图中没有任何指定目标，"box" 为[0,0,0,0]，"confidence" 为 0.0。
        3. 置信度评分：
        - 0.8-1.0：目标存在，且清晰完整。
        - 0.4-0.8：目标存在，但部分遮挡、模糊或存在一定不确定性。
        - 0.0-0.4：目标可能存在。
        4. 完整性：必须包含从 image_id 0 到 {batch_size}-1 的所有图像条目，不得遗漏。
        
        """

        return user_prompt.format(batch_size=self.batch_size, target_desc=self.target_desc)
        # return (
        #     f"你是一个目标检测系统，对于给定的{self.batch_size}张图像，请分别定位图像中可能存在的目标：{self.target_desc}，并给出每个图像中目标的位置和置信度。\\n"
        #     "请严格输出 JSON（不要额外文字），格式如下：\\n"
        #     '{{"image_id":0,"box":[x_min,y_min,x_max,y_max],"confidence":0.0},{"image_id":1,"box":[x_min,y_min,x_max,y_max],"confidence":0.0},...}\\n'
        #     "坐标范围为 0-1000 的整数。\\n"
        #     "confidence 打分规则：\\n"
        #     "1) 若目标对象存在且能被清晰识别，给高置信度 0.8-1.0；\\n"
        #     "2) 若目标对象与描述相似但有不确定性，给中置信度 0.4-0.8；\\n"
        #     "3) 若只能推断目标存在但无法精确定位，给低置信度 0.0-0.4。"
        #     "若目标不存在，box 必须输出 [0,0,0,0]，confidence 必须输出 0.0。\\n"
        #     "注意：每个输入图像都需要输出一个结果，不要漏掉任何一张图像。\\n"
        # )

    @staticmethod
    def _parse_json_output(text: str) -> tuple[np.ndarray, float]:
        text = text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return np.zeros(4, dtype=np.float32), 0.0

        try:
            data = json.loads(match.group(0))
        except Exception:
            return np.zeros(4, dtype=np.float32), 0.0

        box = data.get("box", [0, 0, 0, 0])
        conf = data.get("confidence", 0.0)

        if not isinstance(box, list) or len(box) != 4:
            box = [0, 0, 0, 0]

        box_arr = np.array(box, dtype=np.float32)
        conf_f = float(conf)
        conf_f = max(0.0, min(1.0, conf_f))
        return box_arr, conf_f

    @staticmethod
    def _denorm_xyxy(box_1000: np.ndarray, width: int, height: int) -> np.ndarray:
        if np.allclose(box_1000, np.zeros(4, dtype=np.float32)):
            return np.zeros(4, dtype=np.float32)

        x1, y1, x2, y2 = box_1000.tolist()
        x1 = int(max(0, min(width - 1, (x1 / 1000.0) * width)))
        y1 = int(max(0, min(height - 1, (y1 / 1000.0) * height)))
        x2 = int(max(0, min(width - 1, (x2 / 1000.0) * width)))
        y2 = int(max(0, min(height - 1, (y2 / 1000.0) * height)))

        if x2 <= x1:
            x2 = min(width - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(height - 1, y1 + 1)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def infer_frame(self, frame_path: str, frame_idx: int) -> DetectionResult:
        image = Image.open(frame_path).convert("RGB")
        prompt = self._build_prompt()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device if self.device == "cuda" else self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        box_1000, conf = self._parse_json_output(output_text)
        w, h = image.size
        box_xyxy = self._denorm_xyxy(box_1000, w, h)

        return DetectionResult(
            frame_idx=frame_idx,
            frame_path=frame_path,
            box_xyxy=box_xyxy,
            confidence=conf,
            raw_text=output_text,
        )

    # def infer_batch(self, frame_paths: Sequence[str], frame_indices: Sequence[int]) -> List[DetectionResult]:
    #     if len(frame_paths) != len(frame_indices):
    #         raise ValueError("frame_paths 与 frame_indices 长度不一致")
    #     outputs: List[DetectionResult] = []
    #     for path, idx in zip(frame_paths, frame_indices):
    #         outputs.append(self.infer_frame(path, idx))
    #     return outputs

    def infer_batch(self, frame_paths: Sequence[str], frame_indices: Sequence[int]) -> List[DetectionResult]:
        if not frame_paths:
            return []

        # 1. 加载所有图片
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        
        # 2. 构建多图 Message 内容
        # Qwen3-VL 接收格式通常是图像和文本交替或图像序列后跟指令
        content = []
        for i, img in enumerate(images):
            # 这里的 image_id 对应 prompt 中的顺序
            content.append({"type": "image", "image": img})
        
        # 加入具体的检测指令
        content.append({"type": "text", "text": self._build_prompt()})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # 3. 预处理与推理
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # 注意：这里 images 传的是列表，processor 会自动处理多图 token
        inputs = self.processor(text=[text_prompt], images=images, return_tensors="pt")
        
        device = self.model.device if self.device == "cuda" else self.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens
            )

        # 4. 解析输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # 5. 解析 JSON 数组
        return self._parse_multi_json_output(output_text, frame_paths, frame_indices, images)

    def _parse_multi_json_output(self, text, paths, indices, pil_images) -> List[DetectionResult]:
        results = []
        # 使用正则提取所有 {} 块，或者尝试直接 json.loads 整个列表
        # 考虑到模型可能输出 [{...}, {...}]
        text = text.strip()
        try:
            # 寻找最外层的 JSON 数组结构
            match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
            if match:
                data_list = json.loads(match.group(0))
            else:
                # 备选：如果模型没写方括号，尝试寻找所有的 {}
                items = re.findall(r"\{[^{}]*\}", text)
                data_list = [json.loads(i) for i in items]
        except Exception as e:
            print(f"JSON 解析失败: {e}\n原始输出: {text}")
            data_list = []

        # 将解析结果映射回原始帧
        # 注意：需确保模型输出的顺序/数量与输入一致
        for i in range(len(paths)):
            # 查找对应的 image_id，如果找不到则取索引 i
            item = next((d for d in data_list if d.get("image_id") == i), {})
            if not item and i < len(data_list):
                item = data_list[i]

            box_raw = item.get("box", [0, 0, 0, 0])
            conf = float(item.get("confidence", 0.0))
            
            w, h = pil_images[i].size
            box_xyxy = self._denorm_xyxy(np.array(box_raw), w, h)

            results.append(DetectionResult(
                frame_idx=indices[i],
                frame_path=paths[i],
                box_xyxy=box_xyxy,
                confidence=conf
            ))
        return results