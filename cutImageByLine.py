import cv2
import numpy as np


def detect_lines_and_split_image(image_path, output_dir):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 50, 250, apertureSize=3)
    cv2.imwrite(f"{output_dir}/edges_lines.png", edges)
    # 使用Hough变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=2)

    # 创建一个用于绘制检测到的线条的副本
    line_image = np.copy(image)

    # 存储所有水平线和垂直线的坐标
    horizontal_lines = []
    vertical_lines = []

    # 绘制检测到的线条
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10:  # 水平线
                horizontal_lines.append((x1, y1, x2, y2))
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif abs(x1 - x2) < 10:  # 垂直线
                vertical_lines.append((x1, y1, x2, y2))
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 保存包含检测到的线条的图像
    cv2.imwrite(f"{output_dir}/detected_lines.png", line_image)

    # 根据检测到的线条切分图像
    h_segments = sorted(set([y1 for _, y1, _, _ in horizontal_lines]))
    v_segments = sorted(set([x1 for x1, _, _, _ in vertical_lines]))

    # 如果没有检测到线条，返回
    if not h_segments and not v_segments:
        print("No lines detected for splitting.")
        return

    # 添加图像边界到分割点
    h_segments = [0] + h_segments + [image.shape[0]]
    v_segments = [0] + v_segments + [image.shape[1]]

    # 切分图像并保存
    part = 0
    # for i in range(len(h_segments) - 1):
    #     for j in range(len(v_segments) - 1):
    #         crop_img = image[h_segments[i]:h_segments[i + 1], v_segments[j]:v_segments[j + 1]]
    #         cv2.imwrite(f"{output_dir}/segment_{part}.png", crop_img)
    #         part += 1
    for i in range(len(h_segments) - 1):
        if h_segments[i + 1] - h_segments[i] > 30:
            crop_img = image[h_segments[i]:h_segments[i + 1], :]
            cv2.imwrite(f"{output_dir}/segment_{part}.jpg", crop_img)
            part += 1
        else:
            pass


# 示例使用
image_path = r'D:\data\data\0\temp.jpg'
output_dir = r'D:\data\data\0\save'
detect_lines_and_split_image(image_path, output_dir)
