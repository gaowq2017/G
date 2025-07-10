import os
import yaml
from PIL import Image


image_path = r'D:\data\data\auto'
image_save_path = r'D:\data\data\save'


def image_cut(config):
    file_list = os.listdir(image_path)
    for file in file_list:
        filedir = os.path.join(image_path, file)
        if os.path.isdir(filedir):
            continue
        image = Image.open(filedir)
        for defect in config['pencil']['defectName']:
            result_path = image_save_path + '/' + defect
            os.makedirs(result_path, exist_ok=True)
            count = 0
            for idx in config['pencil'][defect]:
                result_save_path = os.path.join(result_path, str(count) + file)
                x, y, w, h = idx
                image_new = image.crop((x, y, x + w, y + h))
                image_new.save(result_save_path, quality=92)
                count += 1


if __name__ == '__main__':
    config_Path = r'config.yml'
    with open(config_Path) as fp:
        config = yaml.load(fp)
    image_cut(config)

