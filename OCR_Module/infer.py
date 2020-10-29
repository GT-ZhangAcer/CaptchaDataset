# Author: Acer Zhang
# Datetime: 2020/10/12 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

from train import *
from reader import InferReader

DATA_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/sample_img"
CHECKPOINT_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/OCR_Module/output/10"
BATCH_SIZE = 32


def ctc_decode(text, blank=10):
    """
    简易CTC解码器
    :param text: 待解码数据
    :param blank: 分隔符索引值
    :return: 解码后数据
    """
    result = []
    cache_idx = -1
    for char in text:
        if char != blank and char != cache_idx:
            result.append(char)
        cache_idx = char
    return result


if __name__ == '__main__':
    model = pp.Model(Net(is_infer=True), inputs=input_define)

    model.load(CHECKPOINT_PATH)
    model.prepare()

    infer_reader = InferReader(DATA_PATH)
    img_names = infer_reader.get_names()
    results = model.predict(infer_reader, batch_size=BATCH_SIZE)
    index = 0
    for text_batch in results[0]:
        for prob in text_batch:
            out = ctc_decode(prob, blank=10)
            print(f"文件名：{img_names[index]}，推理结果为：{out}")
            index += 1
