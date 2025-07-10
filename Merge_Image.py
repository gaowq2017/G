import cv2
from PIL import Image

# 读取两张图像
image1 = cv2.imread('./hebing/5/1.jpg')
image2 = cv2.imread('./hebing/5/2.jpg')
imageL = Image.open('./hebing/1/1.jpg')
imageR = Image.open('./hebing/1/2.jpg')

# 创建图像拼接器
stitcher = cv2.Stitcher_create()
stitcher.setPanoConfidenceThresh(0)
stitcher.setWaveCorrection(True)
# resize 使两幅图像有统一高度
h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]
image2 = cv2.resize(image2, (h1, w1))
imageR = imageR.resize((w1, h1))
# 进行图像拼接
result = stitcher.stitch((image1, image2))

# 检查拼接是否成功
if result[0] == 0:
    # 拼接成功
    stitched_image = result[1]
    cv2.imwrite('merged_image.jpg', stitched_image)
else:
    # crop_image2 = imageL[0:h1, 0:1750]
    crop_image2 = imageR.crop((170, 0, w1, h1))
    crop_image2.save('crop.jpg')
    result = Image.new('RGB', (w1 + w2 - 170, h1))
    result.paste(imageL, (0, 0))
    result.paste(crop_image2, (w1, 0))
    # cv2.imwrite('merged_image.jpg', result)
    result.save('merged_image.jpg')
    # 拼接失败
    print("图像拼接失败")
