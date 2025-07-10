import cv2
import numpy as np


def load_and_preprocess_image(image_path):
    # 加载图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 二值化图像
    _, binary = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)  # 文字是白色，背景是黑色
    return binary


def find_contours(binary_image):
    # 查找图像中的轮廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def match_shapes(contour1, contour2):
    # 使用Hu矩匹配轮廓
    match_score = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
    return match_score


def match_text_shapes(image1_path, image2_path):
    # 加载并预处理图像
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    image_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image_blur1 = cv2.GaussianBlur(image_gray1, (5, 5), 0)
    image_blur2 = cv2.GaussianBlur(image_gray2, (5, 5), 0)
    # 二值化图像
    _, binary1 = cv2.threshold(image_blur1, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, binary2 = cv2.threshold(image_blur2, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 查找轮廓
    contours1, _ = cv2.findContours(binary1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 对轮廓按照面积进行排序
    sorted_contours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
    sorted_contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)
    # 选择最大的几个轮廓
    num_max_contours = 3  # 指定要选择的最大轮廓数量
    max_contours1 = sorted_contours1[:num_max_contours]
    max_contours2 = sorted_contours2[:num_max_contours]

    cv2.drawContours(image1, max_contours1, -1, (0, 255, 0), 2)
    cv2.drawContours(image2, max_contours2, -1, (0, 255, 0), 2)
    # 保存轮廓图
    cv2.imwrite('temp11.jpg', image1)
    cv2.imwrite('temp22.jpg', image2)
    # 进行形状匹配，匹配每个轮廓
    total_score = 0
    count = 0
    for c1 in max_contours1:
        for c2 in max_contours1:
            score = match_shapes(c1, c2)
            total_score += score
            count += 1

    # 计算平均相似度得分
    average_score = total_score / count if count > 0 else float('inf')
    return average_score


# 指定图像路径
image1_path = r'D:\data\data\FGtemplate\MRYM3ABA_FXCD_1_FGL_2_020824.png'
image2_path = r'D:\data\data\FGtemplate\MRYM3ABA_FXCD_1_FGL_2_020824.png'

# 进行形状匹配
similarity = match_text_shapes(image1_path, image2_path)
print(f'Average shape similarity score: {similarity}')
