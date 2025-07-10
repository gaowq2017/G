import coremltools as ct
import onnx


def convert_onnx_to_coreml(onnx_model_path, coreml_model_path):
    # 加载ONNX模型
    onnx_model = onnx.load(onnx_model_path)

    # 转换为Core ML模型
    coreml_model = ct.converters.onnx.convert(onnx_model, source='onnx')

    # 保存Core ML模型
    coreml_model.save(coreml_model_path)
    print(f"Core ML model saved at {coreml_model_path}")


# 示例使用
onnx_model_path = r'E:\PaddleOCR\checkpoint\save\OCRV4_rec.onnx'
coreml_model_path = r'E:\PaddleOCR\checkpoint\save/model.mlmodel'
convert_onnx_to_coreml(onnx_model_path, coreml_model_path)
