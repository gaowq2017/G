# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests
#
# # load image from the IAM database (actually this model is meant to be used on printed text)
# imagePath = r'D:\data\data\0\1.jpg'
# image = Image.open(imagePath).convert("RGB")
#
# processor = TrOCRProcessor.from_pretrained(r'E:\trocr-large-printed')
# model = VisionEncoderDecoderModel.from_pretrained(r'E:\trocr-large-printed')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values
#
# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print('OK')
from pyzbar import pyzbar
from PIL import Image

def decode_barcode(image_path):
    # 打开图像文件
    image = Image.open(image_path)

    # 使用pyzbar解码条形码
    barcodes = pyzbar.decode(image)

    # 遍历识别到的条形码
    for barcode in barcodes:
        # 提取条形码的边界框位置
        (x, y, w, h) = barcode.rect
        print(f"Barcode found at ({x}, {y}, {w}, {h})")

        # 解码条形码数据
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type
        print(f"Barcode Type: {barcode_type}, Data: {barcode_data}")

if __name__ == "__main__":
    # 替换为您的图像文件路径
    image_path = r"D:\data\data\0\t.jpg"
    decode_barcode(image_path)