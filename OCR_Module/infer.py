# Author: Acer Zhang
# Datetime: 2020/10/12 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

from train import *
from reader import InferReader
from ctc import decode

DATA_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/sample_img"
CHECKPOINT_PATH = "/Users/zhanghongji/PycharmProjects/CaptchaDataset/OCR_Module/output/10"
BATCH_SIZE = 32

if __name__ == '__main__':
    model = pp.Model(Net(is_infer=True), inputs=input_define)

    model.load(CHECKPOINT_PATH)
    model.prepare()

    infer_reader = InferReader(DATA_PATH)
    img_names = infer_reader.get_names()
    results = model.predict(infer_reader, batch_size=BATCH_SIZE)
    index = 0
    for result in results[0]:
        for prob in result:
            out, _ = decode(prob, blank=10)
            print(f"文件名：{img_names[index]}，推理结果为：{out}")
            index += 1
