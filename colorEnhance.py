import cv2
import numpy as np


def adjust_saturation(image_path, factor=3.0):
    # 读取图像
    img = cv2.imread(image_path)
    # research = img[1400: 4000, 600: 4500]
    # cv2.imwrite(r"D:\data\data\0\logo_save\SKY\temp.jpg", research)
    if img is None:
        raise FileNotFoundError("图像文件未找到或无法读取")

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 分离通道
    h, s, v = cv2.split(hsv)

    # 调整饱和度通道
    s = np.clip(s * factor, 0, 255).astype(np.uint8)

    # 合并通道并转回BGR
    hsv_enhanced = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    return result


# 使用示例
input_path = r"D:\data\data\0\logo\FOV2C.jpg"
output_img = adjust_saturation(input_path, factor=2.0)
cv2.imwrite(r"D:\data\data\0\logo_save\SKY\result2.jpg", output_img)
