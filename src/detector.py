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
    category: str = ""
    start_frame_id: int | None = None
    end_frame_id: int | None = None


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

    def _build_prompt(
        self,
        batch_size: int,
        target_desc: str | None = None,
        video_context: str | None = None,
    ) -> str:
        if target_desc is None:
            target_desc = self.target_desc
        video_context = video_context.strip() if isinstance(video_context, str) else ""
        video_context_block = ""
        if video_context:
            video_context_block = f"""
        # Video-level Temporal Context
        下面是一段由完整视频生成的客观时序描述。它只用于补充单帧图像缺失的动作过程和对象交互信息，不是检测标签，也不是答案。
        
        {video_context}
        """
            
        user_prompt = """
        # Role
        你是一个高精度的视觉目标检测系统，需要对多张来自同一视频的图像进行目标定位。

        
        # Task
        给定 {batch_size} 张图像，请分别在每张图像中检测是否存在"Detection Categories"中定义的目标。
        若存在，输出对应目标的"category"、"box"和"confidence"；若不存在，则输出空数组 []。
        
        ## category
        指目标的类别名称。必须是"Detection Categories"包含的目标类别。

        ## box
        指目标的空间位置坐标。坐标使用 0-1000 的标准化整数，顺序为[x_min,y_min,x_max,y_max]。

        ## confidence
        confidence 是类别正确性与定位准确性的综合评分。
        - 0.90-1.00：目标类别明确，关键视觉证据清晰完整，box紧密覆盖完整目标。
        - 0.80-0.89：目标类别明确，box基本准确，但存在轻微遮挡、模糊、边界偏差或背景冗余。
        - 0.60-0.79：目标疑似存在，但关键证据不足，或box不够准确。
        - 0.40-0.59：目标疑似存在，但关键证据不足且box不够准确。
        - <0.40：目标不存在，不要输出。
        若类别不确定或box无法可靠确定，优先输出 []，不要猜测。

        
        # Detection Categories（不同目标类别用“，”分隔）
        {target_desc}


        # Output Rules
        * 每个图像中最多检测2个目标，每个目标的输出包含"category"、"box"、"confidence"。
        * 若目标不存在或不确定，优先输出 []，不要猜测。
        * 必须包含 image_id 从 0 到 {batch_size}-1 的所有项
        * 不要因为Video-level Temporal Context提到某种动作过程，就在当前图像中猜测不可见目标。

        
        # Output Format
        严格输出 JSON 数组，不包含任何解释或额外文本。
        例如：
        [
        {{
            "image_id": 0,
            "detections": [
                {{
                    "category": "类别名称",
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": "0.93"
                }},
                {{
                    "category": "类别名称",
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": "0.45"
                }},
            ]
        }},
        {{
            "image_id": 1,
            "detections": []
        }},
        {{
            "image_id": 2,
            "detections": [
                {{
                    "category": "类别名称",
                    "box": [xmin, ymin, xmax, ymax],
                    "confidence": "0.90"
                }}
        }},
        ...
        ]
        
        # """
        # 指目标检测与定位的置信度分数。评分规则如下：
        # - 0.90-1.00：目标类别明确，关键视觉证据清晰完整，定位准确。
        # - 0.40-0.60：目标疑似存在，但关键证据不足。
        # 注意：置信度分数只能在0.40-0.60或0.90-1.00范围之内；检测和定位结果越可靠，分数越高。
        
    #    confidence 是类别正确性与定位准确性的综合评分。
    #     - 0.90-1.00：目标类别明确，关键视觉证据清晰完整，box紧密覆盖完整目标。
    #     - 0.80-0.89：目标类别明确，box基本准确，但存在轻微遮挡、模糊、边界偏差或背景冗余。
    #     - 0.60-0.79：目标疑似存在，但关键证据不足，或box不够准确。
    #     - 0.40-0.59：目标疑似存在，但关键证据不足且box不够准确。
    #     - <0.40：目标不存在，不要输出。
    #     若类别不确定或box无法可靠确定，优先输出 []，不要猜测。

        # # Detection Rules
        # * 对于"滑滑板"类别，必须满足：
        #    - 画面中清楚可见滑板、滑板车、踏板车或轮滑鞋等滑行工具；
        #    普通走路、跑步、脚步模糊、鞋子阴影、腿部姿态像滑行但看不到工具，都不是"滑滑板"，必须输出 []。
        # * 对于"挥舞物品"类别，必须满足：
        #    - 手中清楚可见一个具体工具，且高举挥舞；
        #    空手挥手、打招呼、正常抬手、正常手持雨伞，都不是"挥舞物品"，必须输出 []。
        # * 对于"奔跑"类别，必须满足：
        #    - 画面中清楚可见人物具有奔跑的动作；
        #    普通走路、散步等，都不是"奔跑"，必须输出 []。

        # * 总体原则：宁可漏检，也不要把相似但不满足定义的普通行为输出为目标类别
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

        #  # Confidence Rules
        # 1. 置信度范围：0.0–1.0
        # 2. 评分依据：
        # - 0.8-1.0：存在Detection Categories中定义的目标，且目标清晰、完整、特征明显
        # - 0.5–0.8：存在Detection Categories中定义的目标，但有遮挡或模糊
        # - 0.2-0.5：不确定是否存在Detection Categories中定义的目标
        # - 0.0-0.2：不存在Detection Categories中定义的类别

        return user_prompt.format(
            batch_size=batch_size,
            target_desc=target_desc,
            video_context_block=video_context_block,
        )

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
        video_context: str | None = None,
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
        content.append({
            "type": "text",
            "text": self._build_prompt(
                batch_size=len(images),
                target_desc=target_desc,
                video_context=video_context,
            ),
        })

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

    @staticmethod
    def _parse_category(det: dict) -> str:
        category = det.get("category", det.get("name", ""))
        if not isinstance(category, str):
            return ""
        return category.strip()

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
                    category="",
                ))
                continue

            for det in detections:
                box_1000, conf = self._parse_box_and_conf(det)
                category = self._parse_category(det)
                box_xyxy = self._denorm_xyxy(box_1000, w, h)
                results.append(DetectionResult(
                    frame_idx=indices[i],
                    frame_path=paths[i],
                    box_xyxy=box_xyxy,
                    confidence=conf,
                    category=category,
                ))
        return results
