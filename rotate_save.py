
import cv2
import os
import random


imagePath = r'D:\data\data\zhuanzhou\1-5ms.bmp'
save_path = r'D:\data\data\zhuanzhou\save'
for i in range(20):
    angle = random.randint(1, 90)
    savePath = save_path + '/' + str(i) + '.bmp'
    # savePath = os.path.join(save_path, file)
    img = cv2.imread(imagePath)
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite(savePath, rotated_image)
