import json
import re
from dataclasses import dataclass
from typing import List, Sequence

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
        stage1_max_new_tokens: int = 128,
        use_two_stage: bool = False,
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
        self.stage1_max_new_tokens = stage1_max_new_tokens
        self.use_two_stage = use_two_stage
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

    def _build_prompt(self, batch_size: int, target_desc: str | None = None) -> str:
        if target_desc is None:
            target_desc = self.target_desc
        user_prompt = """
        # Role
        你是一个高精度的视觉目标检测系统，需要对多张来自同一视频的图像进行目标定位。

        # Task
        给定 {batch_size} 张图像，请在每张图像中检测Detection Categories中定义的目标，并输出对应的边界框（box）和置信度（confidence）。

        # Detection Categories
        {target_desc}

        # Detection Rules
        1. 每张图像独立检测，但应参考其他图像。
        2. 对于多人交互（如打斗、追逐），需分别检测每个参与者。
  
        # Box Rules
        坐标使用 0-1000 的标准化整数，顺序为[x_min,y_min,x_max,y_max]。

        # Confidence Rules
        1. 置信度范围：0.0–1.0
        2. 评分依据：
        - 0.8-1.0：存在Detection Categories中定义的目标，且目标清晰、完整、特征明显
        - 0.5–0.8：存在Detection Categories中定义的目标，但有遮挡或模糊
        - 0.2-0.5：不确定是否存在Detection Categories中定义的目标
        - 0.0-0.2：不存在Detection Categories中定义的类别

        # Output Format
        严格输出 JSON 数组，不包含任何解释或额外文本：

        [
        {{
            "image_id": 0,
            "detections": [
                {{
                    "name": "类别名称",
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": 0.92
                }},
                {{
                    "name": "类别名称",
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": 0.88
                }}
            ]
        }},
        {{
            "image_id": 1,
            "detections": [
                {{
                    "name": "类别名称",
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": 0.6
                }}
            ]
        }},
        ...
        ]
        
        # Output Constraints
        1. 必须包含 image_id 从 0 到 {batch_size}-1 的所有项
        2. 每个 detection 必须包含 name、box、confidence
        3. 若无目标，输出 "detections": []
        
        """
        #请分别分析输入的 {batch_size} 张图像，并定位其中包含的指定目标实例（box），同时给出相应的置信度分数（confidence）。
        # 置信度评分规则：
        # - 0.8-1.0：符合定义的目标存在，且清晰完整。
        # - 0.4-0.8：符合定义的目标存在，但部分遮挡、模糊或存在一定不确定性。
        # - 0.0-0.4：符合定义的目标的存在概率很小。
        # 单张图像中可能存在多个目标，均需定位。

        # # Detection Categories
        # 需要定位的目标类别如下：
        # {target_desc}

        # # Output Format
        # 严格输出 JSON 数组，不包含任何 Markdown 代码块标签或解释文字。格式如下：
        # [
        # {{
        #     "image_id": 0,
        #     "detections": [
        #         {{
        #             "box": [xmin, ymin, xmax, ymax],
        #             "confidence": 0.95
        #         }},
        #         {{
        #             "box": [xmin, ymin, xmax, ymax],
        #             "confidence": 0.88
        #         }}
        #     ]
        # }},
        # {{
        #     "image_id": 1,
        #     "detections": [
        #         {{
        #             "box": [xmin, ymin, xmax, ymax],
        #             "confidence": 0.4
        #         }}
        #     ]
        # }},
        # {{
        #     "image_id": 2,
        #     "detections": []
        # }}
        # ]

        # # Strict Rules
        # 1. 坐标规范：使用 0-1000 的标准化整数，顺序为[x_min,y_min,x_max,y_max]。
        # 2. 多目标处理：如果同一图像中存在多个符合目标描述的实体，必须全部输出。
        # 3. 交互事件处理：对于打斗、追逐、抢夺等多人交互事件，应分别对多个参与者进行定位。
        # 4. 无目标处理：如果图中没有任何指定目标，"detections" 输出空数组 []。
        # 5. 完整性：必须包含从 image_id 0 到 {batch_size}-1 的所有图像条目，不得遗漏。
        
        return user_prompt.format(batch_size=batch_size, target_desc=target_desc)

    def _build_stage1_prompt(self, batch_size: int) -> str:
        # user_prompt = """
        # # Role
        # 你是一个视觉目标类别识别系统。请综合分析输入的 {batch_size} 张视频帧，判断这个视频中存在的目标类别并给出原因，可能存在多种目标类别。

        # # Categories
        # 目标类别如下：
        # {target_desc}

        # # Output Format
        # 严格输出 JSON 对象，不包含任何 Markdown 代码块标签或解释文字。格式如下：
        # {{
        #     "categories": [
        #         {{"name": "打斗", "reason": "多人推搡挥拳"}},
        #         {{"name": "追逐", "reason": "一人快速追赶另一人"}}
        #     ]
        # }}

        # # Rules
        # 1. name表示目标类别，只输出定义的目标类别，不要扩展或改写类别名称。
        # 2. reason表示原因，用简短中文描述，不超过 20 个字。
        # 2. 如果没有任何异常类别，categories 输出空数组 []。
        
        # """
        user_prompt = """
            # Role
            你是一个视频行为理解系统，需要基于多帧图像进行整体时序分析，而不是逐帧独立判断。

            # Task
            给定 {batch_size} 张连续视频帧，请判断视频中是否存在以下目标类别，并给出简要依据。

            # Categories
            目标类别如下（仅允许从中选择，禁止扩展或改写名称）：
            {target_desc}

            # Judgement Rules
            1. 必须基于多帧之间的“连续变化”进行判断，而不是单帧静态信息。
            2. 只有当某类别在多个帧中具有明显一致性特征时，才允许输出。
            3. 若仅在个别帧出现或不确定，请不要输出该类别。
            4. 相似类别需严格区分，避免重复或冲突判断。
            5. 每个类别最多输出一次。

            # Reason Rules
            1. reason必须基于可观察的画面证据，不允许主观猜测（如“可能”、“疑似”）。
            2. 使用简短中文描述（≤20字）。
            3. 应体现关键动作或关系（如“多人持续互相推搡”）。

            # Output Format
            严格输出 JSON 对象，不包含任何解释、Markdown、或多余文本：

            {{
                "categories": [
                    {{"name": "类别1", "reason": "原因"}},
                    {{"name": "类别2", "reason": "原因"}}
                ]
            }}

            # Output Constraints
            1. 仅输出上述 JSON 结构
            2. categories 不得重复类别
            3. 若没有符合条件的类别，输出：
            {{"categories": []}}

        """
        return user_prompt.format(batch_size=batch_size, target_desc=self.target_desc)

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

    def infer_batch(
        self,
        frame_paths: Sequence[str],
        frame_indices: Sequence[int],
        target_desc: str | Sequence[str] | None = None,
    ) -> List[DetectionResult]:
        if not frame_paths:
            return []
        if len(frame_paths) != len(frame_indices):
            raise ValueError("frame_paths 与 frame_indices 长度不一致")
        if target_desc is None:
            target_desc = self.target_desc
        elif not isinstance(target_desc, str):
            target_desc = "，".join(str(cat) for cat in target_desc)

        # 1. 加载所有图片
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        
        # 2. 构建多图 Message 内容
        # Qwen3-VL 接收格式通常是图像和文本交替或图像序列后跟指令
        content = []
        for i, img in enumerate(images):
            # 这里的 image_id 对应 prompt 中的顺序
            content.append({"type": "image", "image": img})
        
        # 加入具体的检测指令
        content.append({"type": "text", "text": self._build_prompt(batch_size=len(images), target_desc=target_desc)})

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

    def detect_categories_with_reason(self, frame_paths: Sequence[str]) -> list[dict]:
        return self._stage1_detect_categories(frame_paths)

    def _stage1_detect_categories(self, frame_paths: Sequence[str]) -> list[dict]:
        if not frame_paths:
            return []

        images = [Image.open(p).convert("RGB") for p in frame_paths]
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": self._build_stage1_prompt(batch_size=len(images))})

        messages = [{"role": "user", "content": content}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=images, return_tensors="pt")
        device = self.model.device if self.device == "cuda" else self.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.stage1_max_new_tokens,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_stage1_categories(output_text)

    @staticmethod
    def _parse_stage1_categories(text: str) -> list[dict]:
        text = text.strip()
        try:
            data = json.loads(text)
        except Exception:
            data_list = QwenVLDetector._load_json_array(text)
            if data_list and isinstance(data_list[0], dict):
                data = data_list[0]
            else:
                data = {}

        categories = data.get("categories", [])
        results: list[dict] = []
        if isinstance(categories, list):
            for item in categories:
                if isinstance(item, dict):
                    name = item.get("name")
                    reason = item.get("reason", "")
                    if isinstance(name, str) and name:
                        results.append({"name": name.strip(), "reason": str(reason)[:50]})
                elif isinstance(item, str) and item:
                    results.append({"name": item.strip(), "reason": ""})
        return results

    # def _stage2_locate(
    #     self,
    #     frame_paths: Sequence[str],
    #     frame_indices: Sequence[int],
    #     target_desc: str,
    # ) -> List[DetectionResult]:
    #     images = [Image.open(p).convert("RGB") for p in frame_paths]
    #     content = [{"type": "image", "image": img} for img in images]
    #     content.append({"type": "text", "text": self._build_prompt(batch_size=len(images), target_desc=target_desc)})

    #     messages = [{"role": "user", "content": content}]
    #     text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    #     inputs = self.processor(text=[text_prompt], images=images, return_tensors="pt")

    #     device = self.model.device if self.device == "cuda" else self.device
    #     inputs = {k: v.to(device) for k, v in inputs.items()}

    #     with torch.inference_mode():
    #         generated_ids = self.model.generate(
    #             **inputs,
    #             max_new_tokens=self.max_new_tokens,
    #         )

    #     generated_ids_trimmed = [
    #         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    #     ]
    #     output_text = self.processor.batch_decode(
    #         generated_ids_trimmed,
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=False,
    #     )[0]

    #     return self._parse_multi_json_output(output_text, frame_paths, frame_indices, images)

    @staticmethod
    def _load_json_array(text: str) -> list:
        text = text.strip()
        try:
            data = json.loads(text)
            return data if isinstance(data, list) else []
        except Exception:
            pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, list) else []
            except Exception:
                pass

        items = re.findall(r"\{[^{}]*\}", text)
        data_list = []
        for item in items:
            try:
                data_list.append(json.loads(item))
            except Exception:
                continue
        return data_list

    @staticmethod
    def _normalize_detection_item(item: dict) -> list[dict]:
        if not isinstance(item, dict):
            return []
        detections = item.get("detections")
        if isinstance(detections, list):
            return [det for det in detections if isinstance(det, dict)]
        if "box" in item or "confidence" in item:
            return [item]
        return []

    @staticmethod
    def _parse_box_and_conf(det: dict) -> tuple[np.ndarray, float]:
        box = det.get("box", [0, 0, 0, 0])
        if not isinstance(box, list) or len(box) != 4:
            box = [0, 0, 0, 0]

        try:
            box_arr = np.array(box, dtype=np.float32)
        except Exception:
            box_arr = np.zeros(4, dtype=np.float32)

        try:
            conf = float(det.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        return box_arr, conf

    def _parse_multi_json_output(self, text, paths, indices, pil_images) -> List[DetectionResult]:
        results = []
        data_list = self._load_json_array(text)
        if not data_list:
            print(f"JSON 解析失败或结果为空，原始输出: {text}")

        # 将解析结果映射回原始帧
        for i in range(len(paths)):
            item = next(
                (d for d in data_list if isinstance(d, dict) and d.get("image_id") == i),
                {},
            )
            if not item and i < len(data_list):
                item = data_list[i]

            detections = self._normalize_detection_item(item)
            w, h = pil_images[i].size
            if not detections:
                results.append(DetectionResult(
                    frame_idx=indices[i],
                    frame_path=paths[i],
                    box_xyxy=np.zeros(4, dtype=np.float32),
                    confidence=0.0,
                ))
                continue

            for det in detections:
                box_1000, conf = self._parse_box_and_conf(det)
                box_xyxy = self._denorm_xyxy(box_1000, w, h)
                results.append(DetectionResult(
                    frame_idx=indices[i],
                    frame_path=paths[i],
                    box_xyxy=box_xyxy,
                    confidence=conf,
                ))
        return results
