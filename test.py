import os
import torch
import numpy as np
import cv2
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# --- 1. 配置参数 ---
checkpoint = "./checkpoints/sam2.1_hiera_large.pt"  # 确保你已下载模型权重
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
video_dir = "test_videos/Explosion008_x264"               # 视频帧文件夹（存放 00000.jpg, 00001.jpg...）
output_dir = "./output_masks/Explosion008_x264"        # 结果保存路径
ann_frame_idx = 70                   # 只有这一帧有标注
ann_obj_id = 1                       # 目标的唯一ID
# 初始 Bounding Box 坐标: [x1, y1, x2, y2]
input_box = np.array([0, 14, 241, 110], dtype=np.float32) 

os.makedirs(output_dir, exist_ok=True)

# --- 2. 初始化模型 ---
predictor = build_sam2_video_predictor(model_cfg, checkpoint)
inference_state = predictor.init_state(video_path=video_dir)

# --- 3. 添加初始 Bounding Box 提示 ---
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=input_box,
)

# --- 4. 以标注帧为中心，前后双向追踪到所有帧 ---
# video_segments 存储所有帧的结果: {frame_idx: {obj_id: mask_np}}
video_segments = {}
num_frames = inference_state["num_frames"]

print("正在进行双向追踪（覆盖全部帧）...")

# 4.1 从标注帧向后（未来帧）追踪
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state,
    start_frame_idx=ann_frame_idx,
    max_frame_num_to_track=num_frames,
    reverse=False,
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# 4.2 从标注帧向前（历史帧）追踪
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state,
    start_frame_idx=ann_frame_idx,
    max_frame_num_to_track=num_frames,
    reverse=True,
):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# --- 5. 可视化并保存结果 ---
def save_masked_frame(frame_path, masks, save_path):
    img = cv2.imread(frame_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    for obj_id, mask in masks.items():
        # 生成随机颜色掩码
        mask = mask.squeeze()
        color = np.array([30, 144, 255]) # 可以改成随机颜色
        h, w = mask.shape[-2:]
        
        # 制作半透明叠加
        mask_image = np.zeros((h, w, 3), dtype=np.uint8)
        mask_image[mask] = color
        img = cv2.addWeighted(img, 1.0, mask_image, 0.5, 0)
        
        # 画出边缘线（可选）
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 255, 255), 2)

    final_img = Image.fromarray(img)
    final_img.save(save_path)

# 遍历所有帧进行保存
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
]
frame_names.sort()

print(f"正在保存结果至: {output_dir}")
for i, frame_name in enumerate(frame_names):
    if i in video_segments:
        frame_path = os.path.join(video_dir, frame_name)
        save_path = os.path.join(output_dir, f"masked_{frame_name}")
        save_masked_frame(frame_path, video_segments[i], save_path)

# 清理内存
predictor.reset_state(inference_state)
print("任务完成！")