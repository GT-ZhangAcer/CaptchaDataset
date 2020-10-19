# Author: Acer Zhang
# Datetime: 2020/10/19 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle as pp

from reader import Reader

# 运行设备选择
USE_GPU = False
# Data文件夹所在路径 - 推荐在自己电脑上使用时选择绝对路径写法 例如:"C:\DLExample\Paddle_classify\data" 可以有效避免因路径问题读不到文件
DATA_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/Classify_Dataset"
# 分类数量，0~9个数字，所以是10分类任务
CLASSIFY_NUM = 10
# CHW格式 - 通道数、高度、宽度
IMAGE_SHAPE_C = 3
IMAGE_SHAPE_H = 30
IMAGE_SHAPE_W = 15


# 定义网络结构 - 需要从pp.nn.Layer类中继承
class Net(pp.nn.Layer):
    def __init__(self, is_infer: bool = False):
        super().__init__()
        self.is_infer = is_infer
        self.layer1 = pp.nn.Linear(in_features=IMAGE_SHAPE_C * IMAGE_SHAPE_H * IMAGE_SHAPE_W, out_features=100)
        self.layer2 = pp.nn.Linear(in_features=100, out_features=CLASSIFY_NUM)

    # 定义网络结构的前向计算过程
    def forward(self, x):
        layer1 = pp.nn.functional.relu(self.layer1(x))
        layer2 = self.layer2(layer1)
        if self.is_infer:
            layer2 = pp.tensor.argmax(layer2)
        return layer2


# 定义输出层 - img的shape应为NCHW格式（并行数量一般为自适应所以为-1，通道数，高，宽）
input_define = pp.static.InputSpec(shape=[-1, IMAGE_SHAPE_C * IMAGE_SHAPE_H * IMAGE_SHAPE_W],
                                   dtype="float32",
                                   name="img")

if __name__ == '__main__':
    label_define = pp.static.InputSpec(shape=[-1, 1],
                                       dtype="int64",
                                       name="label")
    # 实例化网络对象并定义优化器等训练逻辑
    model = pp.Model(Net(), inputs=input_define, labels=label_define)
    # 此处使用SGD优化器，可尝试使用Adam，收敛效果更好更快速
    optimizer = pp.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 损失函数使用交叉熵，评价指标使用准确率
    # 其中Top-k表示推理得到的概率分布中，概率最高的前k个推理结果中是否包含正确标签，如果包含则视为正确，这里的1，2，3分别计算k为1~3的情况
    model.prepare(optimizer=optimizer,
                  loss=pp.nn.CrossEntropyLoss(),
                  metrics=pp.metric.Accuracy(topk=(1, 2)))

    # 这里的reader是刚刚已经定义好的，代表训练和测试的数据
    model.fit(train_data=Reader(data_path=DATA_PATH),
              eval_data=Reader(data_path=DATA_PATH, is_val=True),
              batch_size=32,
              epochs=5,
              save_dir="output/",
              save_freq=10,
              log_freq=1000,
              shuffle=True)
