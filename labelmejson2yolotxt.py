import ujson
import os


######json2instanceSegmentation
jsonDirPath = r'D:\data\bmc\json'
saveDirPath = r'D:\data\bmc\save'
json_list = os.listdir(jsonDirPath)
class_name = ['screw']
for jsonfile in json_list:
    jsonName = os.path.splitext(jsonfile)[0]
    txtName = jsonName + '.txt'
    savePath = os.path.join(saveDirPath, txtName)
    jsonPath = os.path.join(jsonDirPath, jsonfile)
    yolo_obj_list = []
    labels = []
    with open(jsonPath, 'r') as f:
        json_data = ujson.load(f)

    shapes = json_data["shapes"]
    with open(savePath, 'w') as f:
        for shape in shapes:
            line_content = []  # 初始化一个空列表来存储每个形状的坐标信息
            line_content.append(str(class_name.index(shape['label'])))  # 添加类别索引
            # 添加坐标信息
            for point in shape["points"]:
                x = point[0] / json_data["imageWidth"]
                y = point[1] / json_data["imageHeight"]
                line_content.append(str(x))
                line_content.append(str(y))
            # 使用空格连接列表中的所有元素，并写入文件
            f.write(" ".join(line_content) + "\n")
