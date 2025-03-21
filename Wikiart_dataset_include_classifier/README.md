# 使用说明

1.先下载 wikiart 数据集

```
git clone https://huggingface.co/datasets/huggan/wikiart
```

2.解压后放入 Wikiart_dataset 目录下，将 data 文件夹更名为 raw_data  
3.运行 data_process.py 文件，脚本会自动将数据集按照以下格式排布(相较于之前增加了 validation_set)

**DataStructure**  
wikiart/  
├── train/  
│ ├── images/  
│ │ ├── 00010-of-00072/image0.jpg......  
│ │ ├── 00011-of-00072/image0.jpg......  
│ │ └── ...  
│ └── labels/  
│ | ├── 00010-of-00072.json  
│ | ├── 00011-of-00072.json  
│ | └── ...  
└── test/  
| ├── images/  
| │ ├── 00000-of-00072/image0.jpg......  
| │ ├── 00001-of-00072/image0.jpg......  
| │ └── ...  
| └── labels/  
| | ├── 00000-of-00072.json  
| | ├── 00001-of-00072.json  
| | └── ...  
└── val/  
| ├── images/  
| │ ├── 00005-of-00072/image0.jpg......  
| │ ├── 00006-of-00072/image0.jpg......  
| │ └── ...  
| └── labels/  
| | ├── 00005-of-00072.json  
| | ├── 00006-of-00072.json  
| | └── ...  
4.运行 Dataloader_Demo.py 文件，获得数据集中的数据（shuffle 一定要等于 True）

# WikiartDataset.py

构建了 WikiartDataset 类

# Dataloader_Demo.py

提供简易的提取数据 demo
image 为 tensor 类型的图像数据  
label 为类别对应数字  
（num_classes": 11, "names": ["abstract_painting", "cityscape", "genre_painting", "illustration", "landscape", "nude_painting", "portrait", "religious_painting", "sketch_and_study", "still_life", "Unknown Genre"]）

# train_models

里面包括了训练的模型的文件、存储的训练结果（acc_history 文件夹）、训练过程中模型的损失变
化图像。

## 训练

```
# 通过以下代码开始训练:
python train_models/wikiart_train_model_DualPathResNetUNet-18.py
```

在 checkpoint 里面可以找到之前训练好的模型。（超参数在文件名里）正确率大概可以到 67%
