import cv2
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics.pairwise import cosine_similarity

input_path = r'D:\data\data\0\NA_rotated.jpg'
target_path = r'D:\data\data\0\NA.jpg'

img1 = cv2.imread(input_path)
img2 = cv2.imread(target_path)

grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
cv2.imwrite(r'D:\data\data\0\temp1.jpg', grayA)
cv2.imwrite(r'D:\data\data\0\temp2.jpg', grayB)
# _, binary1 = cv2.threshold(grayA, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# _, binary2 = cv2.threshold(grayB, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# hist1 = cv2.calcHist([binary1], [0], None, [2], [0, 256])
# hist2 = cv2.calcHist([binary2], [0], None, [2], [0, 256])

# # 归一化直方图
# hist1 /= hist1.sum()
# hist2 /= hist2.sum()
# ssim_value = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
# # 找到图像的轮廓
# contours1, _ = cv2.findContours(binary1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours2, _ = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # 计算Hu矩
# hu_moments1 = cv2.HuMoments(cv2.moments(contours1[0])).flatten()
# hu_moments2 = cv2.HuMoments(cv2.moments(contours2[0])).flatten()
#
# # ssim_value = np.sum(np.abs(-np.sign(hu_moments1) * np.log10(np.abs(hu_moments1)) -
# #                                - np.sign(hu_moments2) * np.log10(np.abs(hu_moments2))))
# ssim_value = 1.0 - np.sum(np.abs(hu_moments1 - hu_moments2))
# # ssim_value, _ = structural_similarity(binary1, binary2, full=True)

# 初始化SIFT特征提取器
orb = cv2.ORB_create()
# 检测关键点和计算描述符
keypoints1, descriptors1 = orb.detectAndCompute(grayA, None)
keypoints2, descriptors2 = orb.detectAndCompute(grayB, None)

# 如果没有找到任何特征，则返回0相似度
if descriptors1 is None or descriptors2 is None:
    similarity = 0.0

# 创建一个Brute Force匹配器，使用Hamming距离来匹配ORB的二进制描述符
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 根据距离排序匹配项
matches = sorted(matches, key=lambda x: x.distance)

# 如果匹配项不足，返回0相似度
if len(matches) < 4:
    similarity = 0.0

# 提取匹配的关键点坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 使用RANSAC算法计算变换矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 计算内点（inliers）的比例，作为相似度的度量
matches_mask = mask.ravel().tolist()
inliers = sum(matches_mask)
similarity = inliers / len(matches_mask)

print(similarity)

