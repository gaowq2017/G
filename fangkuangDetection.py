import cv2
import os
import numpy as np


def is_contour_a_box(contour, epsilon=0.02):
    # 逼近轮廓以获取多边形
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon * peri, True)
    # 如果逼近结果是四边形，则可能是方框
    return len(approx) == 4


def find_largest_box(image_path, resultsave):
    # 加载图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height = gray.shape[0]
    width = gray.shape[1]
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # cv2.imwrite(r'D:\data\data\images\temp.jpg', thresh)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_box = None
    max_area = 0

    for contour in contours:
        if is_contour_a_box(contour):
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_box = contour

    if largest_box is not None:
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [largest_box], -1, 255, thickness=cv2.FILLED)
        # intersection = cv2.bitwise_and(thresh, mask)
        box_area = cv2.contourArea(largest_box)
        # intersection_area = cv2.countNonZero(intersection)
        intersection_area = height * width
        ritio = box_area / intersection_area


        # 如果方框的面积和交集的面积接近，则认为方框是完整的
        if ritio > 0.6:
            cv2.drawContours(image, [largest_box], -1, (0, 255, 0), 2)  # 绿色表示完整
        else:
            cv2.drawContours(image, [largest_box], -1, (0, 0, 255), 2)  # 红色表示不完整

    # 显示结果
    cv2.imwrite(resultsave, image)
    # cv2.imshow('Largest Box', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# 调用函数
imagePath = r'D:\data\data\images\NG'
savePath = r'D:\data\data\images\NG_save'
imagelist = os.listdir(imagePath)
for file in imagelist:
    filepath = os.path.join(imagePath, file)
    resultsave = os.path.join(savePath, file)
    find_largest_box(filepath, resultsave)


