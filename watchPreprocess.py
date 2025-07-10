import cv2
import os


image_path = r'D:\data\data\BU2AOIimage_5.23\5.26\ng_1'
template_path = r'D:\data\data\BU2AOIimage_5.23\Template\Template.jpg'
save_path = r'D:\data\data\BU2AOIimage_5.23\5.26\save'
for item in os.listdir(image_path):
    filename = item.split('.jpg')[0]
    dir_path = os.path.join(image_path, item)
    src = cv2.imread(dir_path)
    template = cv2.imread(template_path)
    for i in range(4):
        for j in range(3):
            search_image = src[i*1000+0:(i+1)*1000+0, j*940+0:(j+1)*940+0]
            # search_image = src[i * 1000 + 0:(i + 1) * 1000 + 0, j * 980 + 140:(j + 1) * 1000 + 160]
            savePath = save_path + r'\\' + filename + str(i) + '_' + str(j) + '.jpg'
            cv2.imwrite(savePath, search_image)
            res = cv2.matchTemplate(search_image, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            watch_start = (max_loc[0], max_loc[1])
            watch_end = (watch_start[0] + template.shape[1], watch_start[1] + template.shape[0])
            image_watch = search_image[watch_start[1]:watch_end[1], watch_start[0]:watch_end[0], :]
            cv2.imwrite(savePath, image_watch)
            print('OK')

