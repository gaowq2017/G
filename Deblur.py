import cv2


def feature_matching(image1, image2):
    """
    使用特征点匹配比较两幅图像。
    :param image1: 第一幅图像
    :param image2: 第二幅图像
    :return: 匹配的关键点数量
    """
    # 使用 ORB 特征检测器
    orb = cv2.ORB_create()

    # 检测关键点和描述符
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # 创建 BFMatcher 对象
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 匹配描述符
    matches = bf.match(des1, des2)

    # 按照距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    return len(matches), len(kp1), len(kp2)


def normalize_similarity(matches, num_kp1, num_kp2):
    """
    将匹配点数归一化到 0-1 之间。
    :param matches: 匹配的关键点数量
    :param num_kp1: 第一幅图像的关键点数量
    :param num_kp2: 第二幅图像的关键点数量
    :return: 归一化的相似度得分
    """
    max_kp = max(num_kp1, num_kp2)
    if max_kp == 0:
        return 0.0
    similarity = matches / max_kp
    return similarity


# 加载图像
img1 = cv2.imread(r'D:\data\data\0\1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'D:\data\data\0\3.png', cv2.IMREAD_GRAYSCALE)

# 计算特征点匹配
matches, num_kp1, num_kp2 = feature_matching(img1, img2)

# 计算归一化相似度
normalized_similarity = normalize_similarity(matches, num_kp1, num_kp2)
print(f'Normalized similarity score: {normalized_similarity:.2f}')

