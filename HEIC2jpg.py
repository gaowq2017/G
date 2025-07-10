import cv2
import numpy as np
from pillow_heif import register_heif_opener
from PIL import Image
import os

# 注册 HEIF 支持
register_heif_opener()
image_path = r'D:\data\data\long\1\FOV1'
save_path = r'D:\data\data\long\save'
for file in os.listdir(image_path):
    filePath = os.path.join(image_path, file)
    # 打开 HEIC 文件并转换为 RGB
    image = Image.open(filePath).convert("RGB")

    # 转换为 NumPy 数组（OpenCV 格式为 BGR）
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    savePath = os.path.join(save_path, file)
    # 保存为 JPG 格式
    cv2.imwrite(savePath, image_np)
