from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import jieba
from rapidocr_onnxruntime import RapidOCR
from paddleocr import PaddleOCR, draw_ocr
from simhash import Simhash


def getRatated(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(r'D:\data\data\images\temppp.jpg', binary)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [max_contour], 0, (0, 255, 0), 2)
    cv2.imwrite(r'D:\data\data\images\temppp.jpg', image)
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

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated


# img = r"D:\data\data\images\test1.jpg"
# img_ta = r"D:\data\data\images\test.jpg"
# # img = cv2.imread(r"D:\data\data\images\FOV33.jpg")
# # img_ta = cv2.imread(r"D:\data\data\images\FOV22.jpg")
# ocr_engine = RapidOCR()
# result, esp = ocr_engine(img)
# result_Ta, esp_ = ocr_engine(img_ta)
# stringvalue = ''
# stringvalue_Ta = ''
# for txt in result:
#     stringvalue += txt[1].replace(" ", "").upper()
# for str in result_Ta:
#     stringvalue_Ta += str[1].replace(" ", "").upper()

##########更换OCR识别方法
img = cv2.imread(r"D:\data\data\images\FOV11.jpg")
img_ta = cv2.imread(r"D:\data\data\images\FOV22.jpg")
# img_rotated = getRatated(img)
# img_ta_rotated = getRatated(img_ta)
# cv2.imwrite(r'D:\data\data\images\temp.jpg', img_rotated)
# cv2.imwrite(r'D:\data\data\images\tempp.jpg', img_ta_rotated)
gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img_ta, cv2.COLOR_BGR2GRAY)

stringvalue = ''
stringvalue_Ta = ''
ocr = PaddleOCR(use_angle_cls=True, lang="ch")
result = ocr.ocr(gray1, cls=True)
result1 = ocr.ocr(gray2, cls=True)
for line in result[0]:
    stringvalue += line[1][0].replace(" ", "").upper()
for strin in result1[0]:
    stringvalue_Ta += strin[1][0].replace(" ", "").upper()
print(stringvalue)
print(stringvalue_Ta)
# # 创建TF-IDF向量化器
# vectorizer = TfidfVectorizer()


# #######针对中文条件下
# texts = [' '.join(jieba.cut(stringvalue)), ' '.join(jieba.cut(stringvalue_Ta))]
# # 转换文本为TF-IDF向量
# tfidf_matrix = vectorizer.fit_transform(texts)

# ######针对英文条件下
# # 转换文本为TF-IDF向量
# tfidf_matrix = vectorizer.fit_transform([stringvalue, stringvalue_Ta])
#
# # 计算余弦相似度
# similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]


simhash1 = Simhash(stringvalue)
simhash2 = Simhash(stringvalue_Ta)

# 计算哈希之间的汉明距离
distance = simhash1.distance(simhash2)

# 汉明距离与相似度的简单关系（假设最大距离为64位）
similarity = 1 - distance / 64

print(f"Similarity: {similarity:.4f}")

