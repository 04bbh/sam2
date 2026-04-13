import os
import re
import json
import cv2
import numpy as np
import dashscope
from dashscope import MultiModalConversation
from PIL import Image
import matplotlib.pyplot as plt

# 【配置】请在此处填入您的 DashScope API Key
# 建议在系统环境变量中配置 DASHSCOPE_API_KEY，或者直接填在下面
dashscope.api_key = "sk-c012a20df9f141809b5767fa6e620949"

def call_qwen_vl_grounding(image_path, prompt):
    """
    调用 Qwen-VL API 进行视觉定位
    """
    # 构造消息
    # qwen-vl-max 是目前通义千问最强的多模态模型 (通常对应 Qwen2.5-VL/Qwen2-VL 的能力)
    model_name = "qwen3-vl-8b-instruct" 
    
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_path}, # 支持本地路径 "file://..." 或 URL "https://..."
                {"text": prompt}       # 关键：提示词需引导模型输出坐标
            ]
        }
    ]

    print(f"正在请求 {model_name} 进行推理...")
    response = MultiModalConversation.call(model=model_name, messages=messages)

    if response.status_code == 200:
        # content 是一个列表，需要提取文本内容
        content = response.output.choices[0].message.content
        if isinstance(content, list):
            # 提取所有文本内容并连接
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return "".join(text_parts) if text_parts else str(content)
        elif isinstance(content, str):
            return content
        else:
            return str(content)
    else:
        print(f"Error code: {response.code}")
        print(f"Error message: {response.message}")
        return None

def parse_and_visualize(image_path, api_response, output_path="result.jpg"):
    """
    解析 Qwen-VL 返回的特殊格式坐标，并画在图上
    Qwen-VL 返回格式通常包含: <box_2d>[y1, x1, y2, x2]</box_2d>
    坐标是归一化到 [0, 1000] 的整数
    """
    if not api_response:
        return
    
    # 确保 api_response 是字符串
    if isinstance(api_response, list):
        # 如果是列表，提取文本内容
        text_parts = []
        for item in api_response:
            if isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
        api_response = "".join(text_parts) if text_parts else str(api_response)
    elif not isinstance(api_response, str):
        api_response = str(api_response)

    # 1. 读取原始图片获取尺寸
    if image_path.startswith("http"):
        # 如果是URL，这里需要下载图片逻辑，为简化演示，假设是本地路径
        # 实际使用建议先下载图片到本地
        print("可视化仅支持本地文件路径用于绘图。请手动检查文本输出。")
        print(f"API Output: {api_response}")
        return

    # 去掉 file:// 前缀
    local_path = image_path.replace("file://", "")
    img = cv2.imread(local_path)
    if img is None:
        print("无法读取图片")
        return
    
    h_img, w_img = img.shape[:2]

    # 2. 正则提取坐标
    # 尝试多种格式匹配
    patterns = [
        r"<box_2d>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box_2d>",  # 标准格式
        r"<box_2d>\[(\d+),(\d+),(\d+),(\d+)\]</box_2d>",  # 无空格
        r"box_2d>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box_2d>",  # 可能缺少开头的<
        r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]",  # 只有坐标，没有标签
        r"\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",  # 圆括号格式
        r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)",  # 纯数字格式
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
        print("1. API 返回的格式不是预期的 <box_2d> 格式")
        print("2. Prompt 没有引导模型输出坐标")
        print("3. 图像中没有检测到异常")
        # 即使没有匹配到，也保存原图
        cv2.imwrite(output_path, img)
        return

    # 3. 绘制边界框
    for i, box in enumerate(matches):
        # Qwen输出顺序通常是 [y_min, x_min, y_max, x_max] (注意是 y, x, y, x)
        try:
            print(box)
            x1_n, y1_n, x2_n, y2_n = map(int, box)
            
            # 反归一化：将 0-1000 映射回 图像真实尺寸
            x1 = int(x1_n / 1000 * w_img)
            y1 = int(y1_n / 1000 * h_img)
            x2 = int(x2_n / 1000 * w_img)
            y2 = int(y2_n / 1000 * h_img)
            
            # 确保坐标有效
            x1 = max(0, min(x1, w_img - 1))
            y1 = max(0, min(y1, h_img - 1))
            x2 = max(0, min(x2, w_img - 1))
            y2 = max(0, min(y2, h_img - 1))
            
            # 确保 x2 > x1 且 y2 > y1
            if x2 <= x1 or y2 <= y1:
                print(f"⚠️  警告：目标 {i+1} 的坐标无效 (x1={x1}, y1={y1}, x2={x2}, y2={y2})，跳过绘制")
                continue
            
            print(f"目标 {i+1}: 归一化坐标 [{y1_n}, {x1_n}, {y2_n}, {x2_n}] -> 像素坐标 (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            
            # 画框 (BGR 颜色: 红色)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 标号
            cv2.putText(img, f"Obj {i+1}", (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"⚠️  处理目标 {i+1} 时出错: {e}")
            print(f"   原始坐标数据: {box}")
            continue

    # 4. 保存或显示
    cv2.imwrite(output_path, img)
    print(f"可视化结果已保存至: {output_path}")
    
    # 在 Jupyter/Colab 中显示 (可选)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

# --- 主程序入口 ---
if __name__ == "__main__":
    # 示例：假设你有一张名为 'anomaly_frame.jpg' 的本地图片
    # 如果没有，请先找一张图测试，比如有人摔倒、或者有车辆逆行的图
    image_file = "/data/users/wjq/codes/sam2-main/sam2-main/test_videos/Explosion008_x264/070.jpg" 
    
    # 确保文件存在，或者替换为你的图片路径
    if not os.path.exists(image_file):
        # 创建一个假图片用于代码跑通演示
        dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.rectangle(dummy_img, (100, 100), (200, 200), (255, 255, 255), -1) # 白色方块模拟异常
        cv2.imwrite(image_file, dummy_img)
        print("未找到图片，已生成测试图 test_image.jpg")

    # 构造本地文件协议路径
    img_path_input = f"file://{os.path.abspath(image_file)}"

    # 【核心 Prompt 策略】
    # 策略 1：显式定位 (Explicit Grounding) - 你知道异常是什么
    prompt_explicit = """请给出图像中”爆炸（含爆炸后的烟雾）“的空间坐标,请以 [y_min, x_min, y_max, x_max] 格式（数值范围 0-1000）返回 bounding box"""
    
    # # 策略 2：基于描述的异常定位 (Language-guided Anomaly) - 模拟 Training-free VAD
    # # 这种方式测试模型是否理解 'anomaly' 或 'unusual'
    # prompt_anomaly = "Locate the object that looks unusual or stands out in this scene."

    # 执行调用
    result_text = call_qwen_vl_grounding(img_path_input, prompt_explicit)
    
    # 可视化
    parse_and_visualize(image_file, result_text)