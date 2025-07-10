import os
import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import coremltools


class Model_ProdIdx1(nn.Module):
    def __init__(self, model_path):
        super(Model_ProdIdx1, self).__init__()
        self.model = torch.jit.load(model_path)

    def forward(self, inputs):
        x = self.model(inputs)
        x = x.argmax(dim=1)
        return x


class Model_ProdIdx2(nn.Module):
    def __init__(self, model_path):
        super(Model_ProdIdx2, self).__init__()
        self.model = torch.jit.load(model_path)

    def forward(self, inputs):
        x = self.model(inputs)
        x = x.argmax(dim=1)
        return x


def build(config_path, model_Path):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global config
    global model1
    global model2
    with open(config_path) as fp:
        config = yaml.load(fp)
    model1 = Model_ProdIdx1(model_Path)
    model2 = Model_ProdIdx2(model_Path)
    model1.eval()
    # model1.to(device)
    model2.eval()
    # model2.to(device)
    return "Model have loaded!"


def infer(imagePath, prodIdx):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = Image.open(imagePath).convert('RGB')
    results = []
    if prodIdx == 1:
        for defect in config['product1']['defectName']:
            if defect == 'yipian':
                imageArray = imageProcess(image, config['product1'][defect])
                idx = 0
                for img in imageArray:
                    # img = img.to(device)
                    result = model1(img)
                    if result == 0:
                        results.append(config['product1'][defect][idx])
                    idx += 1
            if defect == 'zhuti':
                imageArray = imageProcess(image, config['product1'][defect])
                idx = 0
                for img in imageArray:
                    # img = img.to(device)
                    result = model2(img)
                    if result == 0:
                        results.append(config['product1'][defect][idx])
                    idx += 1
    if prodIdx == 2:
        for defect in config['product1']['defectName']:
            if defect == 'yipian':
                imageArray = imageProcess(image, config['product1'][defect])
                idx = 0
                for img in imageArray:
                    # img = img.to(device)
                    result = model1(img)
                    if result == 1:
                        results.append(config['product1'][defect][idx])
                    idx += 1
            if defect == 'zhuti':
                imageArray = imageProcess(image, config['product1'][defect])
                idx = 0
                for img in imageArray:
                    # img = img.to(device)
                    result = model2(img)
                    if result == 1:
                        results.append(config['product1'][defect][idx])
                    idx += 1
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('arialbi.ttf', 32)
    if results != []:
        for roi in results:
            x1, y1, w1, h1 = roi
            roi_rec = (x1, y1, x1+w1, y1+h1)
            draw.rectangle(roi_rec, outline=(255, 0, 0), width=10)
        draw.text((10, 10), 'Result: NG', fill=(255, 0, 0), font=font)
    else:
        draw.text((10, 10), 'Result: OK', fill=(0, 255, 0), font=font)
    filename = os.path.splitext(imagePath)[0]
    save_path = filename + '.png'
    image.save(save_path)


def imageProcess(image, defectKey):
    imageArray = []
    for defectLocation in defectKey:
        x, y, w, h = defectLocation
        image_new = image.crop((x, y, x + w, y + h))
        image_ary = np.array(image_new, dtype=np.float)
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        img = (image_ary - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()
        imageArray.append(img)

    return imageArray


def release():
    del model1, model2
