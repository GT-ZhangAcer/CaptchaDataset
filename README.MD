# 入门级验证码数据集

## 简介
本数据集从河南部分高校网站登录页面中采集取得。数据集中单张四字图像共计10000张，单张单字图像共计2000张，字符均为0~9数字。数据在自动标注后进行了人工审核，可以保证误差在极小范围。
![示例数据](OCR_Dataset/1.jpg)
## 目录结构
```
_____
|-Classify_Dataset  单张单字数据集
|-Classify_Module   单张单字深度学习训练+推理模型Demo
|-OCR_Dataset   单张四字数据集
|-OCR_Module    单张四字深度学习训练+推理模型Demo
|-LICENSE   许可文件
```
## 数据集结构

以`Classify_Dataset`为例

### Classify_Dataset文件结构
```
_____
|-Classify_Module
    |-1.jpg 图像文件
    |-2.jpg 图像文件
    |-...
    |-label_dict.txt    标签文件

```
### 图像文件
15x30 JPEG-24Bit结构

### 标签文件结构
`{str(1.jpg):int(4), str(文件名):int(label), ...}`
> Tips OCR_Dataset的label非int型，而是str(xxxx)格式

### 简单读取示例(Python)
```python
import os
import PIL.Image

DATA_PATH = "数据集所在的路径"

# 读取label字典
with open(os.path.join(DATA_PATH, "label_dict.txt"), "r", encoding="utf-8") as f:
    info = eval(f.read())   # 若此步骤报错，请检查文件读取是否正常

# 遍历读取到的字典
for file_name in info:
    label = info[file_name] # info形如{"1.jpg": 4,...}，所以该代码返回label值
    img = Image.open(os.path.join(DATA_PATH, file_name))    # 读取图像
```


## 关于
本项目为免费开源项目，使用MIT许可证，详细条款在本项目的根目录中。


