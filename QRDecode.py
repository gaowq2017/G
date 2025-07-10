import cv2
import os
import pyzbar.pyzbar as pyzbar


image_path = r'D:\data\data\save\save\segment_0.png'
# template_path = r'D:\data\unitintray_template\template.jpg'

# image_template = cv2.imread(template_path)
# file_list = os.listdir(image_path)
# for file in file_list:
#     filedir = os.path.join(image_path, file)
#     image = cv2.imread(filedir)
#     search_area = (550, 1550, 840, 1840)
#     for i in range(2):
#         search_image = image[search_area[0]:search_area[1], search_area[2]:search_area[3], :]
#         cv2.imwrite('temp.jpg', search_image)
#         res = cv2.matchTemplate(search_image, image_template, cv2.TM_CCOEFF_NORMED)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#         barcode_start = (max_loc[0], max_loc[1])
#         barcode_end = (barcode_start[0] + image_template.shape[1], barcode_start[1] + image_template.shape[0])
#         image_watch = search_image[barcode_start[1]:barcode_end[1], barcode_start[0]:barcode_end[0], :]
#         # cv2.imwrite('temp.jpg', image_watch)
#         image_watch_gray = cv2.cvtColor(image_watch, cv2.COLOR_BGR2GRAY)
#         barcodes = pyzbar.decode(image_watch_gray)
#         # codeinfo = barcodes[0].data.decode('utf-8')
#         image_blur = cv2.GaussianBlur(image_watch_gray, (5, 5), 0)
#         image_binary = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#         barcodes1 = pyzbar.decode(image_binary)
#         # cv2.imwrite('temp.jpg', image_binary)
#         cv2.imwrite('temp.jpg', image_binary)


image = cv2.imread(image_path)
# height = image.shape[0]
# sn_image = image[:int(height/2)]
# pn_image = image[int(height/2):]
# cv2.imwrite('temp.jpg', sn_image)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
barcodes = pyzbar.decode(image_gray)
# codeinfo = barcodes[0].data.decode('utf-8')
image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
image_binary = cv2.adaptiveThreshold(image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('temp.jpg', image_binary)
barcodes1 = pyzbar.decode(image_binary)
print(barcodes1)
