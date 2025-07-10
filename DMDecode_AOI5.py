import cv2
import os
import numpy as np
from pylibdmtx.pylibdmtx import decode


image_path = r'D:\data\data\pp'
crop_image_path = r'D:\iPhone_Plug\AOI5_barcode'
prePrsessNG = r'D:\iPhone_Plug\AOI5_preProcessNG'
prePrsessOK = r'D:\iPhone_Plug\AOI5_preProcessOK'
for item in os.listdir(image_path):
    dir_path = os.path.join(image_path, item)
    # if os.path.isdir(dir_path):
    # file_path = os.path.join(dir_path + '/FOV4', os.listdir(dir_path + '/FOV4')[0])
    src = cv2.imread(dir_path)
    crop_image = src[30:150, 10:370]
    # crop_image = src[1300:1400, 1650:1880]  # 514
    # save_image_name = os.path.join(crop_image_path, item)
    cv2.imwrite('temp.jpg', crop_image)
    # preNG_image_name = os.path.join(prePrsessNG, item)
    # preOK_image_name = os.path.join(prePrsessOK, item)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    barcode_info = decode(src, timeout=500, max_count=1)
    if barcode_info != []:
        print("dmcode information is : \n%s" % barcode_info)
    else:
        img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 3)
        cv2.imwrite('temp.jpg', img)
        barcode_info = decode(img, timeout=500, max_count=1)
        if barcode_info != []:
            print("dmcode information is : \n%s" % barcode_info)
        else:
            pass
            # cv2.imwrite(save_image_name, crop_image)
            # cv2.imwrite(preNG_image_name, img)
    # if os.path.isfile(dir_path):
    #     continue
