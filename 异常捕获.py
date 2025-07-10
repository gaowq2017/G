import traceback
import cv2
import numpy as np

imagepath = r'D:\data\data\0\NA.jpg'
savepath = r'D:\data\data\0\NA_rotated.jpg'


def get_rotate_angle(img):
    # b, g, r = cv2.split(img)
    r = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # reduced_r = r[1137:2937, 200:2700]
    # cv2.imwrite(r'D:\data\data\0\temp.jpg', r)
    # imgrang1 = cv2.inRange(r, 170, 200)
    _, imgrang1 = cv2.threshold(r, 180, 255, cv2.THRESH_BINARY)
    # _, imgrang1 = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imwrite(r'D:\data\data\0\temp.jpg', imgrang1)
    mask2 = np.zeros(r.shape, np.uint8)

    contours, hierarchy = cv2.findContours(imgrang1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(imgrang1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask2, contours, -1, (255, 255, 255), 5)
    # cv2.imwrite(r'D:\data\data\0\temp.jpg', mask2)
    rect = cv2.minAreaRect(max_contour)
    center1, size, angle = rect[0], rect[1], rect[2]
    if angle > 0:
        if angle > 45:
            angle = 90 - angle
    else:
        if angle < -45:
            angle = -90 - angle
        else:
            angle = -angle

    return angle


img = cv2.imread(imagepath)
angle = get_rotate_angle(img)
height, width = img.shape[:2]
center = (width / 2, height / 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
angle1 = get_rotate_angle(rotated_image)

if angle1 < 0.5:
    cv2.imwrite(savepath, rotated_image)
else:
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite(savepath, rotated_image)
# def faulty_function():
#     return 10 / 0  # 这行代码会引发 ZeroDivisionError 异常
#
# def main():
#     try:
#         # 调用可能引发异常的函数
#         image = cv2.imread(imagepath)
#         faulty_function()
#     except Exception as e:
#         # 捕获异常，并将其保存到文件中
#         with open("error_log.txt", "a") as f:
#             f.write("An error occurred:\n")
#             f.write(traceback.format_exc())
#             f.write("\n")
#
# if __name__ == "__main__":
#     main()
