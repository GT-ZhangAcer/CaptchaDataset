# Author: Acer Zhang
# Datetime: 2020/10/19 
# Copyright belongs to the author.
# Please indicate the source for reprinting.
from reader import InferReader
from train import *

if __name__ == '__main__':
    model = pp.Model(Net(is_infer=True), inputs=input_define)
    model.load("/Users/zhanghongji/PycharmProjects/CaptchaDataset/Classify_Module/output/final")
    model.prepare()
    infer_reader = InferReader(DATA_PATH)
    result = model.predict(test_data=infer_reader)[0]

    img_list = infer_reader.get_names()
    img_index = 0
    for mini_batch in result:
        for sample in mini_batch:
            print(f"{img_list[img_index]}的推理结果为:{sample}")
            img_index += 1

