# 使用说明
1.先下载wikiart数据集
```
git clone https://huggingface.co/datasets/huggan/wikiart
```
2.解压后放入Wikiart_dataset目录下，将data文件夹更名为raw_data  
3.运行data_process.py文件，脚本会自动将数据集按照以下格式排布  

**DataStructure**  
wikiart/  
├── train/  
│    ├── images/  
│    │   ├── 00005-of-00072/image0.jpg......  
│    │   ├── 00006-of-00072/image0.jpg......    
│    │   └── ...  
│    └── labels/  
│    |    ├── 00005-of-00072.json  
│    |    ├── 00006-of-00072.json  
│    |    └── ...  
└── test/  
|    ├── images/  
|    │   ├── 00000-of-00072/image0.jpg......    
|    │   ├── 00001-of-00072/image0.jpg......    
|    │   └── ...  
|    └── labels/  
|    |   ├── 00000-of-00072.json  
|    |   ├── 00001-of-00072.json  
|    |   └── ...  
4.运行Dataloader_Demo.py文件，获得数据集中的数据（shuffle一定要等于True）  

# WikiartDataset.py
构建了WikiartDataset类

# Dataloader_Demo.py
提供简易的提取数据demo
image为tensor类型的图像数据  
label为类别对应数字  
（num_classes": 11, "names": ["abstract_painting", "cityscape", "genre_painting", "illustration", "landscape", "nude_painting", "portrait", "religious_painting", "sketch_and_study", "still_life", "Unknown Genre"]）
