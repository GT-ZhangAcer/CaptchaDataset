# Author: Acer Zhang
# Datetime: 2020/9/21
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import os

import PIL.Image as Image
import numpy as np

from paddle.io import Dataset


class Reader(Dataset):
    def __init__(self, data_path, add_label: bool = True, is_val: bool = False):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param add_label: 是否添加label
        :param is_val: 是否为评估模式
        """
        super().__init__()
        self.data_path = data_path
        self.add_label = add_label
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
        if self.add_label:
            label = self.label_list[self.img_list[index]]
            label = np.array([label], dtype="int64")
            return img, label
        else:
            return img

    def print_sample(self, index: int = 0):
        print("文件名", self.img_list[index], "\t标签值", self.label_list[self.img_list[index]])

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    DATA_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/Classify_Dataset"
    Reader(DATA_PATH).print_sample(1)
