import os
import shutil

image_path = r'D:\data\data\watch\OK'
image_save_path = r'D:\data\data\watch\save'
file_list = os.listdir(image_path)
for file in file_list:
    filedir = os.path.join(image_path, file)
    filename = file.split('.jpg')[0]
    txtName = filename + '.txt'
    with open(os.path.join(image_save_path, txtName), 'a') as f:
        f.write('')
