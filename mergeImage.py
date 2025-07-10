import os
import math
import numpy as np
import cv2
import re


@server.route('/MergeImg', methods=['get', 'post'])
def MergeImg(snInPath, snOutPath):
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
            heightNew = int(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
            widthNew = int(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))
            matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
            matRotation[0, 2] += (widthNew - width) / 2 # 重点在这步，目前不懂为什么加这步
            matRotation[1, 2] += (heightNew - height) / 2 # 重点在这步
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

        cv2.imwrite(os.path.join(Merge_Folder, '{}.jpg'.format(SN)), mergeImage, [cv2.IMWRITE_JPEG_QUALITY, 80])
        res = {'msg': 'OK'}
    except:
        res = {'msg': 'NG'}

    return json.dumps(res, ensure_ascii=False)


inputpath = r'D:\data\data\images'
outputpath = r'D:\data\data\save'
originMergeImg(inputpath, outputpath)
