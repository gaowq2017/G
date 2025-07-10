import cv2
import numpy as np

imgA = cv2.imread("./hebing/1/3.jpg")
imgB = cv2.imread("./hebing/1/4.jpg")
img_1 = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
img_2 = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
h, w = img_2.shape[:2]
fill_loc = np.array([[512, 0], [w, 0], [w, h], [512, h]])
fill_loc1 = np.array([[0, 0], [w-512, 0], [w-512, h], [0, h]])
cv2.fillPoly(img_2, [fill_loc], (0, 0, 0))
cv2.fillPoly(img_1, [fill_loc1], (0, 0, 0))
cv2.imwrite('fill.jpg', img_1)
sift = cv2.SIFT_create(nfeatures=2000)
# sift = cv2.
(kp_1, des_1) = sift.detectAndCompute(img_1, None)
(kp_2, des_2) = sift.detectAndCompute(img_2, None)

kp_A = np.float32([kp.pt for kp in kp_1])   # kp.pt为关键点的坐标
kp_B = np.float32([kp.pt for kp in kp_2])
print(kp_A)
bf = cv2.BFMatcher()
# flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
matches = bf.knnMatch(des_1, des_2, k=2)
good_matches = []
good = []
for (m,n) in matches:
    if m.distance < 0.75*n.distance:
        good_matches.append((m.trainIdx, m.queryIdx))  # 训练(模板)图像的特征描述子索引,查询图像的特征描述子索引
        good.append([m])
img = cv2.drawMatchesKnn(imgA, kp_1, imgB, kp_2, good, None, flags=2)

# 得到匹配对的点坐标
if len(good_matches) >= 4:
    ptr_1 = np.float32([kp_A[i] for (_, i) in good_matches])
    ptr_2 = np.float32([kp_B[i] for (i, _) in good_matches])
# 计算视角变换矩阵
(H, status) = cv2.findHomography(ptr_2, ptr_1, cv2.RANSAC, 1)
result = cv2.warpPerspective(imgB, H, (img_1.shape[1]+img_1.shape[1], img_1.shape[0]))
# cv2.imshow("result_1", result)
cv2.imwrite('result_1.jpg', result)
result[0:imgB.shape[0], 0:imgB.shape[1]] = imgA
# cv2.imshow("result_2", img)
cv2.imwrite("result_2.jpg", img)
# cv2.imshow("result_3", result)
cv2.imwrite("result_3.jpg", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
