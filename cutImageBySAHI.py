import os
import cv2


imagePath = r'E:\data\images'
imageSavePath = r'E:\data\images_save'
labelPath = r'E:\data\labels'
labelSavePath = r'E:\data\labels_save'
slice_size = 1024
overlap = int(slice_size * 0.1)
step = slice_size - overlap

fileList = os.listdir(imagePath)
for file in fileList:
    fileName, fileExpand = os.path.splitext(file)
    filePath = os.path.join(imagePath, file)
    image = cv2.imread(filePath)
    height, width = image.shape[:2]
    ####读取label图
    labelFileName = fileName + '.png'
    labelFilePath = os.path.join(labelPath, labelFileName)
    label = cv2.imread(labelFilePath)
    sliceFileNum = 0
    for y in range(0, height, step):
        for x in range(0, width, step):
            x_end = min(x + slice_size, width)
            y_end = min(y + slice_size, height)
            x0 = x_end - slice_size if x_end - x < slice_size else x
            y0 = y_end - slice_size if y_end - y < slice_size else y
            slice_img = image[y0:y0+slice_size, x0:x0+slice_size]
            saveFileName = fileName + '_' + str(sliceFileNum) + fileExpand
            saveFilePath = os.path.join(imageSavePath, saveFileName)
            cv2.imwrite(saveFilePath, slice_img)
            slice_label = label[y0:y0+slice_size, x0:x0+slice_size]
            saveLabelFileName = fileName + '_' + str(sliceFileNum) + '.png'
            saveLabelFilePath = os.path.join(labelSavePath, saveLabelFileName)
            cv2.imwrite(saveLabelFilePath, slice_label)
            sliceFileNum += 1
