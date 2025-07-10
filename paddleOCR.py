import paddle
import numpy as np
import cv2
from paddle.inference import Config
from paddle.inference import create_predictor


# 初始化检测模型
def init_predictor(model_dir):
    config = Config(model_dir + "/inference.pdmodel", model_dir + "/inference.pdiparams")
    # config.set_data_format("NHWC")
    config.disable_gpu()  # 禁用GPU
    config.enable_mkldnn()  # 启用MKL-DNN加速
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor


# 预处理图像
def preprocess(image, target_size):
    h, w, _ = image.shape
    scale = target_size / max(h, w)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.float32)
    padded_image[:resized_image.shape[0], :resized_image.shape[1], :] = resized_image
    return padded_image, scale


# 预测
def predict(predictor, input_tensor, output_tensor, image):
    input_tensor.copy_from_cpu(image)
    predictor.run()
    output = output_tensor.copy_to_cpu()
    return output


# 加载模型
det_model_dir = r"E:\PaddleOCR\checkpoint\ch_PP-OCRv4_det_infer"
det_predictor = init_predictor(det_model_dir)
det_input_names = det_predictor.get_input_names()
det_output_names = det_predictor.get_output_names()
det_input_tensor = det_predictor.get_input_handle(det_input_names[0])
det_output_tensor = det_predictor.get_output_handle(det_output_names[0])

rec_model_dir = r"E:\PaddleOCR\checkpoint\ch_PP-OCRv4_rec_infer"
rec_predictor = init_predictor(rec_model_dir)
rec_input_names = rec_predictor.get_input_names()
rec_output_names = rec_predictor.get_output_names()
rec_input_tensor = rec_predictor.get_input_handle(rec_input_names[0])
rec_output_tensor = rec_predictor.get_output_handle(rec_output_names[0])

# 加载图像
img_path = r'D:\data\data\images\FOV22.jpg'  # 替换为你要识别的图像路径
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
padded_image, scale = preprocess(image, 256)
padded_image = np.transpose(padded_image, (2, 0, 1))
# 文本检测
det_output = predict(det_predictor, det_input_tensor, det_output_tensor, padded_image[np.newaxis, :])
det_output = det_output[0]  # 假设输出是一个包含检测框的数组

# 文本识别
results = []
for box in det_output:
    x1, y1, x2, y2 = box[:4]
    text_image = image[int(y1 / scale):int(y2 / scale), int(x1 / scale):int(x2 / scale)]
    text_image = cv2.resize(text_image, (100, 32))  # 根据CRNN模型的输入尺寸
    text_image = text_image.astype(np.float32) / 255.0
    text_image = np.transpose(text_image, (2, 0, 1))  # HWC to CHW
    text_output = predict(rec_predictor, rec_input_tensor, rec_output_tensor, text_image[np.newaxis, :])
    text_output = ''.join([chr(c) for c in text_output])
    results.append((box, text_output))

# 输出识别结果
for box, text in results:
    print(f"Box: {box}, Text: {text}")
