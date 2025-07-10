import json
import re


# ########读取txt文件创建json
# txtPath = r'D:\data\data\label.txt'
# delimiter_pattern = r" "
# result = {}
# with open(txtPath, 'r', encoding='utf-8') as f:
#     for line_num, line in enumerate(f, 1):
#         # 清理行内容并跳过空行
#         cleaned_line = line.strip()
#         if not cleaned_line:
#             continue
#
#         # 分割列
#         columns = re.split(delimiter_pattern, cleaned_line)
#
#         # 验证列数
#         if len(columns) < 2:
#             print(f"警告：第{line_num}行列数不足，已跳过")
#             continue
#
#         # 提取前两列
#         col1, col2 = columns[0], columns[1]
#         result.update({col1: str(col2)})
#
# # 生成并保存文件
# with open("pcb_label.json", "w", encoding="utf-8") as f:
#     json.dump(result, f, indent=2, ensure_ascii=False)


############自定义json内容创建json文件

data = {
            "modelfilename": r"D:\workspace\Net_geMo_20250415.onnx",
            "protofilename": r"D:\workspace\Net_geMo_20250415.txt",
            "devtype": "GPU",
            "devindex": 0,
            "tasktype": 'CLS',
            "inferprecision": 'FP16',
            "enginetype": 'TRT',
            "backendtype": "CPP",
            "batchsize": 1,
            "imagewidthinput": 200,
            "imageheightinput": 200,
            "imagechannelinput": 3,
            "visualtype": 'NONE',
        }

# 将字典写入JSON文件r
with open(r'D:\data\data\param.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
