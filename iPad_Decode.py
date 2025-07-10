import cv2
from pyzbar.pyzbar import decode


inputPath = r'D:\data\data\iPad\9_target.jpg'
image_iPad = cv2.imread(inputPath)
# image_180 = cv2.flip(image_iPad, -1)
# cv2.imwrite(r'D:\data\data\iPad\temp.jpg', image_180)
iPad_barcode = image_iPad[320:520, 1300:1880, :]  ####精细坐标
# iPad_barcode = image_iPad[120:800, 1000:2000, :]
cv2.imwrite(r'D:\data\data\iPad\temp.jpg', iPad_barcode)
gray = cv2.cvtColor(iPad_barcode, cv2.COLOR_BGR2GRAY)
barcode_info = decode(gray)
if len(barcode_info) != 0:
    info = barcode_info[0].data.decode('utf-8')
else:
    info = ''

print(info)

