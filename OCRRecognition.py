import cv2
import re
import numpy as np
import time
# from rapidocr_onnxruntime import RapidOCR
from rapidocr import RapidOCR


def rotate_image(image, angle):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 计算图像中心
    center = (w // 2, h // 2)

    # 生成旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图像尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 应用仿射变换
    return cv2.warpAffine(image, M, (new_w, new_h))


def get_rotate_angle(img):
    b, g, r = cv2.split(img)
    # reduced_r = r[1137:2937, 200:2700]
    # reduced_r = r[1137:2937, 200:3200]
    imgrang1 = cv2.inRange(r, 10, 120)
    # cv2.imwrite(r"D:\data\data\images\heihei.jpg", imgrang1)
    contours, hierarchy = cv2.findContours(imgrang1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
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


img = cv2.imread(r'D:\data\data\hezi.jpg')
# ang = get_rotate_angle(img)
# img = rotate_image(img, -ang)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (480, 64))
# _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# cv2.imwrite(r"D:\data\data\0\1.jpg", img)
ocr_engine = RapidOCR()
result = ocr_engine(img)
result.vis(r"D:\data\data\vis_result.jpg")
stringvalue = ''
for txt in result.txts:
    stringvalue += txt.replace(" ", "").upper().replace("O", "0")
print(stringvalue)
pattern1 = r'\d+'
ftrtxt1 = re.findall(pattern1, stringvalue)
print(ftrtxt1)


from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=False, lang="ch")  # need to run only once to download and load model into memory
img_path = r'D:\data\data\result_img_0.png'
start = time.perf_counter()
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, binary = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# binary = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
# cv2.imwrite(r"D:\data\data\0\1.jpg", binary)
# image = cv2.resize(image, (480, 64))
result = ocr.ocr(image)

# 显示结果
from PIL import Image
result = result[0]
end = time.perf_counter()
print("运行时间：", end - start)
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
txts = ''
for line in result:
    txts += line[1][0].replace(" ", "").upper()

# pattern = r'\d+'
# ftrtxt = re.findall(pattern, txts)
# ress = txts.find('512')
# resss = txts.find('A1234')
# resssS = txts.find('SPG/12CCPU/19CGPU/16GB/512GB')
# gb = txts.split('CAPACITY')[1].split('NOTE')[0]
# ne = txts.split('NO.')[1].split(',')[0]
# biao = txts.split(',')[1].split('COO')[0]
# partNum = txts[-15:].split('。')[1]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save(r'D:\data\data\OCR\save\result.jpg')

# import easyocr
# # 创建OCR对象
# reader = easyocr.Reader(['en'])
# # 识别文字
# result = reader.readtext(r"D:\data\data\data\temp.jpg")
# # 处理识别结果
# for (text, bbox, confidence) in result:
#     print(f'Text: {text}, Bbox: {bbox}, Confidence: {confidence}')


