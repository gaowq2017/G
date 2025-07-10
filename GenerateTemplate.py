import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import os
import cv2 as cv


def Tensor2Image(img1):
    img1 = torch.squeeze(img1).numpy()
    img1 = img1.transpose((1, 2, 0))
    img1 = np.array(img1 * 255, dtype=np.uint8)
    img2 = Image.fromarray(img1)

    return img2


def regulateposition(img):
    gray = img[:, :, 0]
    edges = cv.Canny(gray, 100, 250)

    cv.imshow('result', edges)
    cv.waitKey(0)
    lines = cv.HoughLines(edges, 1, np.pi / 1440, 500)
    rhos = []
    for line in lines:
        rho, theta = line[0]
        rho = rho
        rhos.append(abs(rho))

    # Determine the most common angle
    rho_counts = np.bincount(np.round(rhos).astype(int))
    count = np.sum(rho_counts)

    return count


def generateTemplate(templatePath):

    File_path = templatePath
    File_name = ['FOV1', 'FOV2', 'FOV3', 'FOV4']
    n1 = 2 ** 2
    f1 = nn.AvgPool2d(n1, stride=n1)
    trans = transforms.Compose([transforms.ToTensor(), ])

    idx1, idx2, idx3, idx4 = [], [], [], []

    cnt = []
    vertect = []
    for i, fn in enumerate(File_name):

        fi = fn[-1]
        img0 = Image.open(File_path + '/' + fn + '.jpg')
        img0 = trans(img0)
        img0 = img0[None]

        _, _, m, n = img0.shape
        Gain = [1, n, m, n, m]

        #### read txt
        df_bw = pd.read_csv(File_path + '/' + fn + '.txt', sep=' ', header=None, names=['class', 'x', 'y', 'w', 'h'],
                            index_col=False)
        df_1 = df_bw[df_bw['class'].isin([1])].reset_index(drop=True)
        df_2 = df_bw[df_bw['class'].isin([2])].reset_index(drop=True)
        df_3 = df_bw[df_bw['class'].isin([0])].reset_index(drop=True)

        df_1 = df_1 * Gain
        df_2 = df_2 * Gain
        df_3 = df_3 * Gain

        # vertect.append([int(df_3.loc[0, 'y']+ df_3.loc[0, 'h']/2),
        #                 int(df_3.loc[0, 'x']+ df_3.loc[0, 'w']/2) if (int(df_3.loc[0, 'x'] + df_3.loc[0, 'w']/2)) < 2900 else int(df_3.loc[0, 'x']- df_3.loc[0, 'w']/2)])

        vertect.append([int(df_3.loc[0, 'y'] + df_3.loc[0, 'h'] / 2),
                        int(df_3.loc[0, 'x'] + df_3.loc[0, 'w'] / 2) if df_3.loc[0, 'x'] < 1512 else int(
                            df_3.loc[0, 'x'] - df_3.loc[0, 'w'] / 2)])

        # print(df_1)
        os.makedirs(File_path + '/' + fn, exist_ok=True)

        cntt = list()

        for j in range(len(df_1)):
            imgt0 = img0[:, :,
                    int(df_1.loc[j, 'y'] - df_1.loc[j, 'h'] / 2):int(df_1.loc[j, 'y'] + df_1.loc[j, 'h'] / 2) + 1, \
                    int(df_1.loc[j, 'x'] - df_1.loc[j, 'w'] / 2):int(df_1.loc[j, 'x'] + df_1.loc[j, 'w'] / 2) + 1]

            cntt.append([float(df_1.loc[j, 'x']), float(df_1.loc[j, 'y'])])

            imgt1 = Tensor2Image(imgt0)
            imgt1.save(File_path + '/' + fn + '/T' + str(j + 1) + '.jpg', quality=100)

            imgt2 = f1(imgt0)
            imgt3 = Tensor2Image(imgt2)
            imgt3.save(File_path + '/' + fn + '/T0' + str(j + 1) + '.jpg', quality=100)

            if j == 0:
                idx1.append(
                    (int(df_2.loc[j, 'y'] - df_2.loc[j, 'h'] / 2), int(df_2.loc[j, 'y'] + df_2.loc[j, 'h'] / 2) + 1))
                idx2.append(
                    (int(df_2.loc[j, 'x'] - df_2.loc[j, 'w'] / 2), int(df_2.loc[j, 'x'] + df_2.loc[j, 'w'] / 2) + 1))
            else:
                idx3.append(
                    (int(df_2.loc[j, 'y'] - df_2.loc[j, 'h'] / 2), int(df_2.loc[j, 'y'] + df_2.loc[j, 'h'] / 2) + 1))
                idx4.append(
                    (int(df_2.loc[j, 'x'] - df_2.loc[j, 'w'] / 2), int(df_2.loc[j, 'x'] + df_2.loc[j, 'w'] / 2) + 1))

        cnt.append(cntt)

    print('idx1 = ', idx1)
    print('idx2 = ', idx2)
    print('idx3 = ', idx3)
    print('idx4 = ', idx4)

    print('Cnt = np.array(', cnt, ')')

    X = vertect[0][0]
    Y = vertect[1][0]

    doublesize = 200
    singlesize = 135

    ratio = min((X + Y - doublesize) / 2, X - singlesize, Y - singlesize) / 3072

    # print(ratio)

    ## pattern match vis setting
    Outputsize = [3072, 2208]
    # ratio = 1.05
    bar = 2550

    Cropsize = (np.array(Outputsize) * ratio).astype(int)
    Length = [Cropsize[0], Cropsize[0], Cropsize[0], Cropsize[0]]
    Width = [2 * Cropsize[1] - bar, 2 * Cropsize[1] - bar, bar, bar]
    start = vertect
    order = [[2, 1, 4, 3], [3, 4, 1, 2], [1, 2, 3, 4], [4, 3, 2, 1]]

    Pnt = []
    for i in range(4):
        endx = start[i][0] + Length[i] if (start[i][0] + Length[i]) < 4032 else start[i][0] - Length[i]
        endy = start[i][1] + Width[i] if (start[i][1] + Width[i]) < 3024 else start[i][1] - Width[i]
        arrt = [[start[i][0], start[i][1]], [start[i][0], endy], [endx, endy], [endx, start[i][1]]]
        arrt = [[start[i][1], start[i][0]], [endy, start[i][0]], [endy, endx], [start[i][1], endx]]
        arrt = [arrt[order[i][0] - 1], arrt[order[i][1] - 1], arrt[order[i][2] - 1], arrt[order[i][3] - 1]]
        Pnt.append(arrt)

    print('Pnt = np.array(', Pnt, ')')

    n_point = 5
    Length = [3072, 3072, 3072, 3072]
    Width = [bar, bar, 4416 - bar, 4416 - bar]

    print('n_point = 5')
    print('bar = ', str(int(bar / ratio)))
    print('Length = [', Outputsize[0], ', ', Outputsize[0], ', ', Outputsize[0], ', ', Outputsize[0], ']')
    print('Width  = [', Outputsize[1] * 2, '- bar, ', Outputsize[1] * 2, '- bar, bar, bar]')


if __name__ == '__main__':
    templatePath = r'D:\data\data\iPadTemplate'
    generateTemplate(templatePath)
