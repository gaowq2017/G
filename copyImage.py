import os
import shutil
from PIL import Image

image_path = r'D:\data\data\0725TM\Pass'
image_save_path = r'D:\data\data\images'

# count = 245
# dir_list = os.listdir(image_path)
# for dir in dir_list:
#     dirPath = os.path.join(image_path, dir)
#     fileList = os.listdir(dirPath)
#     for file in fileList:
#         if file == 'positive.jpg':
#         # if file.endswith('.jpg'):
#             filePath = os.path.join(dirPath, file)
#             # image = Image.open(filePath)
#             newFileName = 'positive' + str(count) + '.jpg'
#             newFilePath = os.path.join(image_save_path, newFileName)
#             # image.save(newFilePath)
#             shutil.move(filePath, newFilePath)
#             count += 1

dir_list = os.listdir(image_path)
for dir in dir_list:
    dirPath = os.path.join(image_path, dir)
    dirlist = os.listdir(dirPath)
    for dirl in dirlist:
        if dirl == '.DS_Store':
            continue
        if dirl == 'FOV1':
            tempPath = os.path.join(dirPath, 'FOV1')
        if dirl == 'FOV3':
            tempPath = os.path.join(dirPath, 'FOV3')
        tempdirPath = os.listdir(tempPath)
        for file in tempdirPath:
            # if file.startswith('FOV1'):
            filePath = os.path.join(tempPath, file)
            newFilePath = os.path.join(image_save_path, file)
            shutil.move(filePath, newFilePath)
