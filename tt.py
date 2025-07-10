import cv2
import os
# from pylibdmtx.pylibdmtx import decode
from pyzbar.pyzbar import decode
import pyzbar.wrapper
# from colorCompare import colorFit

image_path = r'D:\data\data\BU2AOIimage_5.23\5.26\ng_5.26'
save_path = r'D:\data\data\BU2AOIimage_5.23\5.26\save'
for file in os.listdir(image_path):
    filename = file.split('.jpg')[0]
    filePath = os.path.join(image_path, file)
    image = cv2.imread(filePath)
    b, g, r = cv2.split(image)
    # _, img_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, img_r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # blured = cv2.GaussianBlur(b, (3, 3), 0)
    _, img_b = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)
    _, img_r = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
    img = img_b - img_r
    # cv2.imwrite(r'D:\data\data\BU2AOIimage_5.23\5.26\save\temp1.jpg', img_b)
    # cv2.imwrite(r'D:\data\data\BU2AOIimage_5.23\5.26\save\temp2.jpg', img_r)
    # cv2.imwrite(r'D:\data\data\BU2AOIimage_5.23\5.26\save\temp3.jpg', img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fileNum = 0
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        area = width * height
        if area < 200000:
            continue
        else:
            # print(area)
            x, y = rect[0]
            # tempimage = image[int(y - height / 2): int(y + height / 2), int(x - width / 2): int(x + width / 2), ]
            if x < 384:
                x = 0
            elif x + 384 > 3024:
                x = 3024 - 768
            else:
                x -= 384
            if y < 384:
                y = 0
            elif y + 384 > 4032:
                y = 4032-768
            else:
                y -= 384
            tempimage = image[int(y): int(y + 768), int(x): int(x + 768), ]
            savefile = filename + '_' + str(fileNum) + '.jpg'
            fileNum += 1
            savePath = os.path.join(save_path, savefile)
            cv2.imwrite(savePath, tempimage)
print('OK')
max_contour = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(max_contour)

print('OK')

# research = b[1500: 4000, 0: 3000]
# cv2.imwrite(r'D:\data\data\0\Result.jpg', research)
# image = cv2.flip(cv2.transpose(img), 0)
# cv2.imwrite(r'D:\data\data\images\Result.jpg', image)
# mirrored_image = cv2.flip(image, 1)
# cv2.imwrite(r'D:\data\data\images\FOV44.jpg', mirrored_image)
# const = cv2.copyMakeBorder(img, 50, 50, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# imagePath = r'D:\data\data\0\FOV2_71.jpg'
# img = cv2.imread(imagePath)
# b, g, r = cv2.split(img)
# research = b[2000: 7000, 600: 4500]
# cv2.imwrite(r'D:\data\data\temp.jpg', research)
# imgrange = cv2.inRange(research, 160, 255)
# cv2.imwrite(r'D:\data\data\temp.jpg', imgrange)
# contours, hierarchy = cv2.findContours(imgrange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# max_contour = max(contours, key=cv2.contourArea)
# rect = cv2.minAreaRect(max_contour)
# width, height = rect[1]
# print(width, height)

imagePath = r'D:\data\data\0\t.jpg'
img = cv2.imread(imagePath)
# img_90 = cv2.flip(cv2.transpose(img), 1)
# cv2.imwrite(r'D:\data\data\iPadTemplate\temp.jpg', img_90)
# reserch_image = img_90[1400:1530, 440:910, :]
# # reserch_image = img_90[1200:1600, 300:1000, :]
# # reserch_image = img_90[1450:1520, 440:910, :]
# cv2.imwrite(r'D:\data\data\iPadTemplate\research.jpg', reserch_image)

resize = cv2.resize(img, (4096, 512))   ###### interpolation=cv2.INTER_CUBIC
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blured = cv2.GaussianBlur(gray, (3, 3), 0)
# cv2.imwrite(r'D:\data\data\iPadTemplate\resize.jpg', blured)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blured)
# cv2.imwrite(r'D:\data\data\0\save\temp.jpg', enhanced)
barcode_info = decode(resize, symbols=[pyzbar.wrapper.ZBarSymbol.CODE128])
if len(barcode_info) != 0:
    print("dmcode information is : \n%s" % barcode_info)
else:
    gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # _, img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    img = cv2.inRange(gray, 120, 255)
    # _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    # img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # img = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(r'D:\data\data\0\save\temp.jpg', img)
    barcode_info = decode(img)
    print('')



