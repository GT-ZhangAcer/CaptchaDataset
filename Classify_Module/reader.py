# Author: Acer Zhang
# Datetime: 2020/9/21
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os

import PIL.Image as Image
import numpy as np

from paddle.io import Dataset


class Reader(Dataset):
    def __init__(self, data_path, is_val: bool = False):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param is_val: 是否为评估模式
        """
        super().__init__()
        self.data_path = data_path
        # 因为数据集中图片的文件名是纯数字形式，这里不必去获取文件夹下的图片，直接使用range生成即可
        self.img_list = [str(i) + ".jpg" for i in range(1, 800)]

        # 打开存放label数据的文件
        with open(os.path.join(data_path, "label_dict.txt"), 'r') as f:
            self.label_list = eval(f.read())

        # 划分数据集 - 该步骤可以重新设计逻辑，传入不同的路径来代表读取的数据集更佳，但因为本次数据集训练集和验证集咋同一个文件夹，故在此进行分割
        self.img_list = self.img_list[:500] if not is_val \
            else self.img_list[500:800]

    def __getitem__(self, index):
        """
        获取一组数据
        :param index: 文件索引号
        :return:
        """
        # 第一步打开图像文件并获取label值
        img_path = os.path.join(self.data_path, self.img_list[index])
        img = Image.open(img_path)
        img = np.array(img, dtype="float32").flatten()
        img /= 255
        label = self.label_list[self.img_list[index]]
        label = np.array([label], dtype="int64")
        return img, label

    def print_sample(self, index: int = 0):
        print("文件名", self.img_list[index], "\t标签值", self.label_list[self.img_list[index]])

    def __len__(self):
        return len(self.img_list)


class InferReader(Dataset):
    def __init__(self, dir_path=None, img_path=None):
        """
        数据读取Reader(推理)
        :param dir_path: 推理对应文件夹（二选一）
        :param img_path: 推理单张图片（二选一）
        """
        super().__init__()
        if dir_path:
            # 获取文件夹中所有图片路径
            self.img_names = [i for i in os.listdir(dir_path) if os.path.splitext(i)[1] == ".jpg"]
            self.img_paths = [os.path.join(dir_path, i) for i in self.img_names]
        elif img_path:
            self.img_names = [os.path.split(img_path)[1]]
            self.img_paths = [img_path]
        else:
            raise Exception("请指定需要预测的文件夹或对应图片路径")

    def get_names(self):
        """
        获取推理文件名顺序
        """
        return self.img_names

    def __getitem__(self, index):
        # 获取图像路径
        file_path = self.img_paths[index]
        # 使用Pillow来读取图像数据并转成Numpy格式
        img = Image.open(file_path)
        img = np.array(img, dtype="float32").flatten() / 255
        return img

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':
    DATA_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/Classify_Dataset"
    Reader(DATA_PATH).print_sample(1)
