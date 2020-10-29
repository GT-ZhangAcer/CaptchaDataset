# Author: Acer Zhang
# Datetime: 2020/10/12 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle as pp

from reader import Reader, InferReader

# 数据集路径设置
DATA_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/OCR_Dataset"
# 训练轮数
EPOCH = 10
# 训练超参数
BATCH_SIZE = 16
# 分类数量设置 - 因数据集中共包含0~9共10种数字+，所以是11分类任务
CLASSIFY_NUM = 11
# CHW格式 - 通道数、宽度、高度
IMAGE_SHAPE_C = 3
IMAGE_SHAPE_H = 30
IMAGE_SHAPE_W = 70

# 需与Reader中保持一致 - 次部分在Notebook中可以合并
# 数据集标签长度最大值
LABEL_MAX_LEN = 4


# 定义网络结构
class Net(pp.nn.Layer):
    def __init__(self, is_infer: bool = False):
        super().__init__()
        self.is_infer = is_infer

        # 定义一层3x3卷积+BatchNorm
        self.conv1 = pp.nn.Conv2D(in_channels=IMAGE_SHAPE_C,
                                  out_channels=32,
                                  kernel_size=3)
        self.bn1 = pp.nn.BatchNorm2D(32)
        # 定义一层步长为2的3x3卷积进行下采样+BatchNorm
        self.conv2 = pp.nn.Conv2D(in_channels=32,
                                  out_channels=64,
                                  kernel_size=3,
                                  stride=2)
        self.bn2 = pp.nn.BatchNorm2D(64)
        # 定义一层1x1卷积压缩通道数，输出通道数设置为比LABEL_MAX_LEN稍大的定值即可
        self.conv3 = pp.nn.Conv2D(in_channels=64,
                                  out_channels=LABEL_MAX_LEN + 4,
                                  kernel_size=1)
        # 定义全连接层，压缩并提取特征（可选）
        self.linear = pp.nn.Linear(in_features=429,
                                   out_features=128)
        # 定义RNN层来更好提取序列特征，此处为双向LSTM输出为2 x hidden_size，可尝试换成GRU等RNN结构
        self.lstm = pp.nn.LSTM(input_size=128,
                               hidden_size=64,
                               direction="bidirectional")
        # 定义输出层，输出大小为分类数
        self.linear2 = pp.nn.Linear(in_features=64 * 2,
                                    out_features=CLASSIFY_NUM)

    def forward(self, ipt):
        # 卷积 + ReLU + BN
        x = self.conv1(ipt)
        x = pp.nn.functional.relu(x)
        x = self.bn1(x)
        # 卷积 + ReLU + BN
        x = self.conv2(x)
        x = pp.nn.functional.relu(x)
        x = self.bn2(x)
        # 卷积 + ReLU
        x = self.conv3(x)
        x = pp.nn.functional.relu(x)
        # 将3维特征转换为2维特征
        x = pp.tensor.flatten(x, 2)
        # 全连接 + ReLU
        x = self.linear(x)
        x = pp.nn.functional.relu(x)
        # 双向LSTM - [0]代表取双向结果，[1][0]代表forward结果,[1][1]代表backward结果，详细说明可参考官方文档LSTM
        x = self.lstm(x)[0]
        # 输出层
        x = self.linear2(x)

        # 若为推理模式则需做softmax获取标签概率
        if self.is_infer:
            x = pp.nn.functional.softmax(x)
            x = pp.tensor.argmax(x, axis=-1)
        return x


# 定义输入数据格式，shape中添加-1则可以在推理时自由调节batch size
input_define = pp.static.InputSpec(shape=[-1, IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W],
                                   dtype="float32",
                                   name="img")

if __name__ == '__main__':
    # 监督训练需要定义label，预测则不需要该步骤
    label_define = pp.static.InputSpec(shape=[-1, 4],
                                       dtype="int32",
                                       name="label")


    class CTCLoss(pp.nn.Layer):
        def __init__(self):
            """
            定义CTCLoss
            """
            super().__init__()

        def forward(self, ipt, label):
            input_lengths = pp.tensor.creation.fill_constant([BATCH_SIZE, 1], "int64", LABEL_MAX_LEN + 4)
            label_lengths = pp.tensor.creation.fill_constant([BATCH_SIZE, 1], "int64", LABEL_MAX_LEN)
            # 按文档要求进行转换dim顺序
            ipt = pp.tensor.transpose(ipt, [1, 0, 2])
            # 计算loss
            loss = pp.nn.functional.ctc_loss(ipt, label, input_lengths, label_lengths, blank=10)
            return loss


    # 实例化模型
    model = pp.Model(Net(), inputs=input_define, labels=label_define)

    # 定义优化器
    optimizer = pp.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())

    # 为模型设置运行环境以及优化策略
    model.prepare(optimizer=optimizer,
                  loss=CTCLoss())

    # 执行训练
    model.fit(train_data=Reader(DATA_PATH),
              eval_data=Reader(DATA_PATH, is_val=True),
              batch_size=BATCH_SIZE,
              epochs=EPOCH,
              save_dir="output/",
              save_freq=1,
              log_freq=100)
