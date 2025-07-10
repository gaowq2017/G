import cv2
import base64
import json


imagePath = r'D:\workspace\images_\gemoNG_(1).bmp'
image = cv2.imread(imagePath)

height, width, channels = image.shape
if channels == 3:
    typeNum = 16
else:
    typeNum = 0
# 2. 编码为 PNG 格式的二进制数据
_, buffer = cv2.imencode('.bmp', image)

# 3. 将二进制数据编码为 base64 字符串
image_base64 = base64.b64encode(buffer).decode('utf-8')

# 4. 包装为 JSON 数据结构
data = {
    "inputdata": [{'data': image_base64, 'rows': height, 'cols': width, 'type': typeNum}]
}

# 5. 保存为 JSON 文件
with open(r'D:\data\data\inputData.json', 'w') as f:
    json.dump(data, f, indent=4)
