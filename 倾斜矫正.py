import cv2
import numpy as np

# 读取图像
image = cv2.imread(r'D:\data\data\0\rotate.jpg')


def getAngle(image):
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # 使用Canny边缘检测
    # _, edges = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, edges = cv2.threshold(blured, 100, 255, cv2.THRESH_BINARY)
    # edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (255, 0, 0), 10)
    # cv2.imwrite(r'D:\data\data\0\save\temp.jpg', image)
    # 找到最大轮廓
    largest_contour = max(contours, key=cv2.contourArea)
    # 计算最小外接矩形
    rect = cv2.minAreaRect(largest_contour)

    # 计算倾斜角度
    angle = rect[2]
    if angle > 0:
        if angle > 45:
            angle = 90 - angle
    else:
        if angle < -45:
            angle = -90 - angle
        else:
            angle = -angle

    return angle


angle = getAngle(image)
# 旋转图像进行校正
h, w = image.shape[:2]
center = (w / 2, h / 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
angle1 = getAngle(rotated)
if angle1 < 0.7:
    pass
else:
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(r'D:\data\data\0\save\Corrected.jpg', rotated)
