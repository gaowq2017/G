import cv2
import os
import numpy as np
from pylibdmtx.pylibdmtx import decode

image_path = r'D:\iPhone_Plug\AOI9'
crop_image_path = r'D:\iPhone_Plug\AOI9_barcode'
prePrsessNG = r'D:\iPhone_Plug\AOI9_preProcessNG'
prePrsessOK = r'D:\iPhone_Plug\AOI9_preProcessOK'
count = 0
for item in os.listdir(image_path):
    dir_path = os.path.join(image_path, item)
    if os.path.isdir(dir_path):
        file_path = os.path.join(dir_path + '/FOV4', os.listdir(dir_path + '/FOV4')[0])
        src = cv2.imread(file_path)
        # crop_image = src[1220:1380, 1650:1950]  # 514
        crop_image = src[1390:1530, 2180:2580]  # 516
        save_image_name = os.path.join(crop_image_path, os.listdir(dir_path + '/FOV4')[0])
        preNG_image_name = os.path.join(prePrsessNG, os.listdir(dir_path + '/FOV4')[0])
        preOK_image_name = os.path.join(prePrsessOK, os.listdir(dir_path + '/FOV4')[0])
        cv2.imwrite(save_image_name, crop_image)
        gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
        barcode_info = decode(gray, timeout=500, max_count=1)
        if barcode_info != []:
            # cv2.imwrite(preOK_image_name, img)
            sn = barcode_info[0].data.decode('utf-8')
            res = {'msg': sn}
            count += 1
            print("dmcode information is : \n%s" % barcode_info)
            # return json.dumps(res, ensure_asici=False)
        else:
            alpha = 1.2  # 调整对比度，大于1增加对比度，小于1降低对比度
            beta = 30  # 调整亮度，正值增加亮度，负值降低亮度
            adjusted_image = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            img = cv2.adaptiveThreshold(adjusted_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)
            barcode_info = decode(img, timeout=500, max_count=1)
            if barcode_info != []:
                sn = barcode_info[0].data.decode('utf-8')
                res = {'msg': sn}
                count += 1
                print("dmcode information is : \n%s" % barcode_info)
            else:
                cv2.imwrite(save_image_name, crop_image)
                cv2.imwrite(preNG_image_name, img)
    if os.path.isfile(dir_path):
        continue
print('Detect numeber is %d', count)
