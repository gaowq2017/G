import cv2
import os
import flask
import json
import math
import re
import numpy as np
from pylibdmtx.pylibdmtx import decode


server = flask.Flask(__name__)


@server.route('/MergeImg', methods=['get', 'post'])
def MergeImg():
    snInPath = flask.request.args.get("inputPath")
    snOutPath = flask.request.args.get("outputPath")
    try:
        SN = os.path.basename(snInPath)
        Merge_Folder = os.path.join(snOutPath, 'Merge')
        os.makedirs(Merge_Folder, exist_ok=True)
        img_list0 = os.listdir(snInPath)
        img_list0 = list(filter(lambda x: x.endswith('.jpg'), img_list0))
        if len(img_list0) < 4:
            return
        mergeImage = np.zeros((4416, 6144, 3), dtype=np.uint8)
        for idx in range(4):
            pattern = 'FOV' + str(idx + 1)
            img_list = list(filter(lambda x: re.search(pattern, x) != None, img_list0))
            if len(img_list) == 0:
                break
            img_file = img_list[0]
            img = cv2.imread(os.path.join(snInPath, img_file))
            img = np.array(img)
            if idx == 0:
                img = img[300:, 200:-150, :]
            if idx == 1:
                img = img[300:, :-200, :]
            if idx == 2:
                img = img[300:, :-200, :]
            if idx == 3:
                img = img[300:, 200:-50, :]

            height, width = img.shape[:2]
            degree = 0
            if idx == 0:
                degree = 270
            elif idx == 1:
                degree = 90
            elif idx == 2:
                degree = 270
            elif idx == 3:
                degree = 90
            # 旋转后的尺寸
            heightNew = int(
                width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
            widthNew = int(
                height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
            matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
            matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
            matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步
            imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
            imgRotation = cv2.resize(imgRotation, (3072, 2208))
            if idx == 0:
                mergeImage[2208:4416, 0:3072, :] = imgRotation
            elif idx == 1:
                mergeImage[2208:4416, 3072:6144, :] = imgRotation
            elif idx == 2:
                mergeImage[0:2208, 0:3072, :] = imgRotation
            elif idx == 3:
                mergeImage[0:2208, 3072:6144, :] = imgRotation
        mergeImage = mergeImage[400:4000, 500:5500, :]
        cv2.imwrite(os.path.join(Merge_Folder, '{}.jpg'.format(SN)), mergeImage, [cv2.IMWRITE_JPEG_QUALITY, 80])
        res = {'msg': 'OK'}
    except:
        res = {'msg': 'NG'}

    return json.dumps(res, ensure_ascii=False)


@server.route('/getiPad', methods=['get', 'post'])
def getiPad():
    inputPath = flask.request.args.get("inputPath")
    outputPath = flask.request.args.get("outputPath")
    templatePath = flask.request.args.get("templatePath")
    IPADTEMPLATE = cv2.imread(templatePath)
    try:
        file = os.path.basename(inputPath)
        image = cv2.imread(inputPath)
        res = cv2.matchTemplate(image, IPADTEMPLATE, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        image_start = (max_loc[0], max_loc[1])
        image_end = (image_start[0] + IPADTEMPLATE.shape[1], image_start[1] + IPADTEMPLATE.shape[0])
        image_iPad = image[image_start[1]:image_end[1], image_start[0]:image_end[0], :]
        image_save_path = os.path.join(outputPath, file)
        cv2.imwrite(image_save_path, image_iPad)
        ##### Detect iPad
        gray = cv2.cvtColor(image_iPad, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        width = gray.shape[1]
        intersection_area = height * width
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_box = None
        max_area = 0
        epsilon = 0.02
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_box = contour

        if largest_box is not None:
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [largest_box], -1, 255, thickness=cv2.FILLED)
            box_area = cv2.contourArea(largest_box)
            ritio = box_area / intersection_area
            if ritio > 0.6:
                res = {'msg': 'OK'}
            else:
                res = {'msg': 'NG'}
        else:
            res = {'msg': 'OK'}
    except:
        res = {'msg': 'TimeOut'}

    return json.dumps(res, ensure_ascii=False)


@server.route('/detipad', methods=['get', 'post'])
def detipad():
    inputPath = flask.request.args.get('inputPath')
    try:
        image = cv2.imread(inputPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height = gray.shape[0]
        width = gray.shape[1]
        intersection_area = height * width
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_box = None
        max_area = 0
        epsilon = 0.02
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(contour)
                if area > max_area:
                    max_area = area
                    largest_box = contour

        if largest_box is not None:
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [largest_box], -1, 255, thickness=cv2.FILLED)
            box_area = cv2.contourArea(largest_box)
            ritio = box_area / intersection_area
            if ritio > 0.6:
                res = {'msg': 'OK'}
            else:
                res = {'msg': 'NG'}
        else:
            res = {'msg': 'OK'}
    except:
        res = {'msg': 'TimeOut'}

    return json.dumps(res, ensure_ascii=False)


@server.route('/getCode', methods=['get', 'post'])
def getCode():
    inputPath = flask.request.args.get("inputPath")
    templatePath = flask.request.args.get("templatePath")
    try:
        image = cv2.imread(inputPath)
        image_template = cv2.imread(templatePath)
        # image_template = CODETEMPLATE
        search_image = image[1200:2200, 1900:2900, :]
        res = cv2.matchTemplate(search_image, image_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        start = (max_loc[0], max_loc[1])
        end = (start[0] + image_template.shape[1], start[1] + image_template.shape[0])
        image_code = search_image[start[1]:end[1], start[0]:end[0], :]
        cv2.imwrite(r'D:\data\data\save\temp\temp.jpg', image_code)
        codeinfo = decode(image_code, timeout=500, max_count=1)
        if len(codeinfo) != 0:
            dminfo = codeinfo[0].data.decode('utf-8')
            res = {'result': 'OK', 'sn': dminfo}
        else:
            gray = cv2.cvtColor(image_code, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, pre_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # pre_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
            cv2.imwrite(r'D:\data\data\save\temp\temp.jpg', pre_image)
            codeinfo = decode(pre_image)
            if len(codeinfo) != 0:
                dminfo = codeinfo[0].data.decode('utf-8')
                res = {'result': 'OK', 'sn': dminfo}
            else:
                pre_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
                codeinfo = decode(pre_image)
                if len(codeinfo) != 0:
                    dminfo = codeinfo[0].data.decode('utf-8')
                    res = {'result': 'OK', 'sn': dminfo}
                else:
                    res = {'result': 'NG', 'sn': ''}
        # json.dumps 序列化时对中文默认使用的ascii编码，输出中文需要设置ensure_ascii=False
        return json.dumps(res, ensure_ascii=False)
    except:
        # json.dumps 序列化时对中文默认使用的ascii编码，输出中文需要设置ensure_ascii=False
        res = {'result': 'TimeOut'}
        return json.dumps(res, ensure_ascii=False)


if __name__ == '__main__':
    server.run(host='127.0.0.1', port=10086)
