import cv2
import numpy as np

# 读取两张图像
image1 = cv2.imread('./hebing/1/1.jpg')
image2 = cv2.imread('./hebing/1/2.jpg')

# 转换为灰度图像
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# h, w = gray2.shape[:2]
# fill_loc = np.array([[1024, 0], [w, 0], [w, h], [1024, h]])
# cv2.fillPoly(gray2, [fill_loc], (0, 0, 0))

# 使用特征匹配器（sift）找到两张图像的关键点和匹配
sift = cv2.ORB_create(nfeatures=2000)
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# 使用FLANN匹配器进行特征匹配
# flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 筛选匹配的特征
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 获取匹配特征的关键点坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用透视变换找到重叠区域
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 获取第一张图像的尺寸
h1, w1 = image1.shape[:2]

# 使用透视变换对第一张图像进行变换
result = cv2.warpPerspective(image1, M, (w1 + image2.shape[1], h1))

# 将第二张图像复制到结果图像的右侧
result[0:image2.shape[0], 0:image2.shape[1]] = image2

# 创建掩模（mask）来去重
mask = np.zeros(result.shape[:2], dtype=np.uint8)
mask[0:image2.shape[0], 0:image2.shape[1]] = 255

# 应用掩模
result = cv2.inpaint(result, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

# 保存拼接后的图像
cv2.imwrite('result.jpg', result)
