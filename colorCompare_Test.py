import cv2


def calculate_histogram(image):
    # 计算图像的直方图
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def colorFit(image, label):
    # 加载两张图像
    res = cv2.matchTemplate(image, label, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    start = (max_loc[0], max_loc[1])
    end = (start[0] + label.shape[1], start[1] + label.shape[0])
    image_logo = image[start[1]:end[1], start[0]:end[0], :]
    # 将图像转换为HSV颜色空间
    image1_hsv = cv2.cvtColor(image_logo, cv2.COLOR_BGR2HSV)
    image2_hsv = cv2.cvtColor(label, cv2.COLOR_BGR2HSV)

    # 计算图像的直方图
    hist1 = calculate_histogram(image1_hsv)
    hist2 = calculate_histogram(image2_hsv)

    # 比较直方图并计算相似度
    similarity = 1.0 - cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return similarity
