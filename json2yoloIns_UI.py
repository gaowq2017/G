import tkinter as tk
from tkinter import filedialog, messagebox
import os
import re
import ujson


class FileRenamerApp:
    def __init__(self, master):
        self.master = master
        master.title("文件批量重命名工具")

        # 源文件夹选择组件
        self.source_frame = tk.Frame(master)
        self.source_frame.pack(pady=5)

        self.source_label = tk.Label(self.source_frame, text="源文件夹:")
        self.source_label.pack(side=tk.LEFT)

        self.source_entry = tk.Entry(self.source_frame, width=50)
        self.source_entry.pack(side=tk.LEFT, padx=5)

        self.source_button = tk.Button(
            self.source_frame,
            text="浏览...",
            command=self.select_source_directory
        )
        self.source_button.pack(side=tk.LEFT)

        # 目标文件夹选择组件
        self.target_frame = tk.Frame(master)
        self.target_frame.pack(pady=5)

        self.target_label = tk.Label(self.target_frame, text="目标文件夹:")
        self.target_label.pack(side=tk.LEFT)

        self.target_entry = tk.Entry(self.target_frame, width=50)
        self.target_entry.pack(side=tk.LEFT, padx=5)

        self.target_button = tk.Button(
            self.target_frame,
            text="浏览...",
            command=self.select_target_directory
        )
        self.target_button.pack(side=tk.LEFT)

        # 处理按钮
        self.process_btn = tk.Button(
            master,
            text="开始处理文件",
            command=self.process_files,
            bg="#4CAF50",
            fg="white"
        )
        self.process_btn.pack(pady=10)

    def select_source_directory(self):
        """选择源文件夹"""
        directory = filedialog.askdirectory()
        if directory:
            self.source_entry.delete(0, tk.END)
            self.source_entry.insert(0, directory)

    def select_target_directory(self):
        """选择目标文件夹"""
        directory = filedialog.askdirectory()
        if directory:
            self.target_entry.delete(0, tk.END)
            self.target_entry.insert(0, directory)

    def process_files(self):
        try:
            """处理文件重命名和保存"""
            source_dir = self.source_entry.get()
            target_dir = self.target_entry.get()
            # 创建目标文件夹（如果不存在）
            os.makedirs(target_dir, exist_ok=True)
            if not source_dir or not target_dir:
                messagebox.showerror("错误", "请先选择源文件夹和目标文件夹")
                return
            class_name = []
            classNamePath = os.path.join(source_dir, 'class_name.txt')
            with open(classNamePath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # 清理行内容并跳过空行
                    cleaned_line = line.strip()
                    if not cleaned_line:
                        continue
                    # 分割列
                    columns = re.split(r',', cleaned_line)
                    for className in columns:
                        class_name.append(className)
            json_list = os.listdir(source_dir)
            file_num = 0
            for jsonfile in json_list:
                if jsonfile.endswith('.json'):
                    jsonName = os.path.splitext(jsonfile)[0]
                    txtName = jsonName + '.txt'
                    savePath = os.path.join(target_dir, txtName)
                    jsonPath = os.path.join(source_dir, jsonfile)
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
                    file_num += 1

            messagebox.showinfo("完成", f"成功处理 {file_num} 个文件！\n" f"已保存到：{target_dir}")
        except Exception as e:
            messagebox.showerror("错误", f"处理文件时发生错误：\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FileRenamerApp(root)
    root.mainloop()
