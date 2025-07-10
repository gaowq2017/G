import cv2
import os
import numpy as np
from pylibdmtx.pylibdmtx import decode


image_path = r'D:\data\watch\17'
template_path = r'D:\data\watch\template.jpg'
save_path = r'D:\data\data\temp.jpg'
for item in os.listdir(image_path):
    dir_path = os.path.join(image_path, item)
    src = cv2.imread(dir_path)
    template = cv2.imread(template_path)
    for i in range(4):
        for j in range(6):
            search_image = src[i*1000+600:(i+1)*1000+600, j*1000+600:(j+1)*1000+600]
            cv2.imwrite(save_path, search_image)
            res = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            watch_start = (max_loc[0], max_loc[1])
            watch_end = (watch_start[0] + template.shape[1], watch_start[1] + template.shape[0])
            image_watch = search_image[watch_start[1]:watch_end[1], watch_start[0]:watch_end[0], :]
            cv2.imwrite(save_path, image_watch)
            barcode_info = decode(image_watch, timeout=500, max_count=1)
            if len(barcode_info) != 0:
                print("dmcode information is : \n%s" % barcode_info)
            else:
                gray = cv2.cvtColor(image_watch, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
                # img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 2)
                cv2.imwrite(save_path, img)
                barcode_info = decode(img, timeout=500, max_count=1)
                if barcode_info != []:
                    print("dmcode information is : \n%s" % barcode_info)
                else:
                    pass
