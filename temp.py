# from transformers import CLIPModel, CLIPProcessor
# import torch
# # 模型名称
# model_name = r"E:\OCR\model\clip-vit-large-patch14"
#
# # 加载 CLIP 模型和处理器
# model = CLIPModel.from_pretrained(model_name)
# processor = CLIPProcessor.from_pretrained(model_name)
# # ONNX 保存路径
# onnx_path = r"E:\OCR\model\clip_vit_l_14.onnx"
#
# # 导出模型
# dummy_input = torch.randn(1, 3, 224, 224)
# torch.onnx.export(
#     model.vision_model,
#     dummy_input,
#     onnx_path,
#     input_names=["pixel_values"],
#     output_names=["image_embeddings"],
#     dynamic_axes={"pixel_values": {0: "batch_size"}},
#     opset_version=13,
# )
# print(f"ONNX model saved to {onnx_path}")

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# 加载 CLIP 模型
model_name = r"E:\clip_Vit_B_32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


# 预处理第一张图像
image1 = Image.open(r"D:\data\data\0\image1.jpg").convert("RGB")

# 预处理第二张图像
image2 = Image.open(r"D:\data\data\0\image2.jpg").convert("RGB")

# 使用 CLIP 处理器对图像进行预处理
inputs = processor(images=[image1, image2], return_tensors="pt", padding=True)

# 提取图像特征
with torch.no_grad():
    image_features = model.get_image_features(**inputs)

# 归一化特征向量
image_features = image_features / image_features.norm(dim=1, keepdim=True)

# 计算余弦相似度
similarity = torch.cosine_similarity(image_features[0:1], image_features[1:2])

print(similarity.item())


