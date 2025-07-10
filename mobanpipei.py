"""
功能：银行卡数字识别
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def Im_show(windos_name, image):
    cv2.imshow(windos_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 将传入的轮廓序列按要求排序输出
# 输入参数：cnts/轮廓；method/排序类型
def Sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # 返回排序好的轮廓和外接矩形
    return cnts, boundingBoxes


# 读入模板图片
template = cv2.imread("images/ocr_a_reference.png")
# im_show("template",template)
# 灰度图
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# im_show("gray_template",gray_template)
# 二值化
bin_template = cv2.threshold(gray_template, 10, 255, cv2.THRESH_BINARY_INV)[1]
# Im_show("bin_template",bin_template)

# 模板轮廓
I, template_contours, hierarchy = cv2.findContours(bin_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(I,type(I))
# r = cv2.drawContours(template.copy(),template_contours,-1,(0,0,255),3)
# Im_show("r",r)
# print(np.array(contours).shape,contours)
# 模板轮廓的外接矩形
template_boundingboxs = [cv2.boundingRect(c) for c in template_contours]
# print(template_boundingboxs,type(template_boundingboxs))
# 将外接矩形按从左到右排序
sort_template_boundingboxs = Sort_contours(template_contours)[1]
# 模板字典
template_dict = {}
for i in range(10):
    (x, y, w, h) = sort_template_boundingboxs[i]
    roi = bin_template[y:y+h, x:x+w]
    print(roi)
    roi = cv2.resize(roi, (54, 85))
    template_dict[i] = roi
    # im_show(str(i),roi)
# print(template_dict)

# 读入待检测银行卡图片
# 图片大小：368X583
target_img = cv2.imread("images/credit_card_01.png")
# Im_show('target_img',target_img)
gray_target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
# gray_target_img = cv2.resize(gray_target_img,(368,583))
# Im_show('gray_target_img',gray_target_img)
bin_target_img = cv2.threshold(gray_target_img, 240, 255, cv2.THRESH_BINARY)[1]
# Im_show('bin_target_img',bin_target_img)

# 初始化卷积核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))

# 闭操作
close_img = cv2.morphologyEx(bin_target_img, cv2.MORPH_CLOSE, kernel)
# Im_show("close_img",close_img)

# 对闭处理后的图像进行轮廓检测找出四个目标轮廓
I, target_cnts, hierarchy = cv2.findContours(close_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
r = cv2.drawContours(target_img.copy(), target_cnts, -1, (0, 0, 255), 1)
# Im_show("r",r)

# 获得目标区域
target_boundingboxs = []
for (i, c) in enumerate(target_cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    d = w/float(h)
    if (d > 3.2 and d<3.7):
        x -= 5
        y -= 5
        w += 10
        h += 10
        target_boundingboxs.append((x, y, w, h))
        cv2.rectangle(target_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
# print(target_boundingboxs)
# Im_show("target_img",target_img)

# 将目标区域的轮廓从左到右排序
target_boundingboxs = sorted(target_boundingboxs,key=lambda x:x[0])
# print(target_boundingboxs)

# 对目标区域进行模板匹配
digits = ''         # 银行卡号
for (m,boundingbox) in enumerate(target_boundingboxs):
    (X, Y, W, H) = boundingbox
    big_roi = bin_target_img[Y:Y+H, X:X+W]
    # Im_show("big_roi",big_roi)
    I, contours, hierarchy = cv2.findContours(big_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_boundingboxs = Sort_contours(contours)[1]
    for (n, roi_boundingbox) in enumerate(roi_boundingboxs):
        (x, y, w, h) = roi_boundingbox
        roi = big_roi[y:y + h, x:x + w]
        roi = cv2.resize(roi, (54, 85))
        # Im_show("roi",roi)
        scores = []
        for i in range(10):
            score = cv2.matchTemplate(template_dict[i], roi, cv2.TM_CCOEFF_NORMED)
            scores.append(score)
        val = scores.index(max(scores))
        digits += str(val)

        # 画出来
        cv2.putText(target_img, str(val), (X+x, Y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
print("卡号为：", digits)
Im_show("result", target_img)
