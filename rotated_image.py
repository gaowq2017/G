import cv2
import os


def get_rotate_angle(img):
    b, g, r = cv2.split(img)
    # reduced_r = r[1137:2937, 200:2700]
    # reduced_r = r[1000:5600, 1600:5200]  ####for DV04
    reduced_r = r[600:5400, 900:4500]
    # cv2.imwrite(r'D:\data\data\auto\temp.jpg', reduced_r)
    imgrang1 = cv2.inRange(reduced_r, 10, 120)
    contours, hierarchy = cv2.findContours(imgrang1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(img, contours, -1, (255, 0, 0), 10)
    # cv2.imwrite(r'D:\data\data\auto\temp.jpg', img)
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


image_path = r'D:\data\data\AutoDecode'
save_path = r'D:\data\data\auto'
result_save_path = r'D:\data\data\result'
template_path = r'D:\data\data\iPad_Template\Template.jpg'
Template = cv2.imread(template_path)
image_list = os.listdir(image_path)
for file in image_list:
    imagePath = os.path.join(image_path, file)
    savePath = os.path.join(save_path, file)
    resultSavePath = os.path.join(result_save_path, file)
    img = cv2.imread(imagePath)
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    angle = get_rotate_angle(img)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    angle1 = get_rotate_angle(rotated_image)
    if angle1 < 0.5:
        cv2.imwrite(savePath, rotated_image)
        res = cv2.matchTemplate(rotated_image, Template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        start = (max_loc[0], max_loc[1])
        end = (start[0] + Template.shape[1], start[1] + Template.shape[0])
        image_code = rotated_image[start[1]:end[1], start[0]:end[0], :]
        cv2.imwrite(resultSavePath, image_code)
    else:
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
        cv2.imwrite(savePath, rotated_image)
        res = cv2.matchTemplate(rotated_image, Template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        start = (max_loc[0], max_loc[1])
        end = (start[0] + Template.shape[1], start[1] + Template.shape[0])
        image_code = rotated_image[start[1]:end[1], start[0]:end[0], :]
        cv2.imwrite(resultSavePath, image_code)

