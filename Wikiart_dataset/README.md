# DataStructure
wikiart/  
├── train/  
│    ├── images/  
│    │   ├── train-00005-of-00072/  
│    │   ├── train-00006-of-00072/  
│    │   └── ...  
│    └── labels/  
│    |    ├── train-00005-of-00072/  
│    |    ├── train-00006-of-00072/  
│    |    └── ...  
└── test/  
|    ├── images/  
|    │   ├── test-00000-of-00072/  
|    │   ├── test-00001-of-00072/  
|    │   └── ...  
|    └── labels/  
|    |   ├── test-00000-of-00072/  
|    |   ├── test-00001-of-00072/  
|    |   └── ...


# WikiartDataset.py
构建了WikiartDataset类

# Dataloader_Demo.py
提供简易的提取数据demo  
image为tensor类型的图像数据  
label为类别对应数字  
（num_classes": 11, "names": ["abstract_painting", "cityscape", "genre_painting", "illustration", "landscape", "nude_painting", "portrait", "religious_painting", "sketch_and_study", "still_life", "Unknown Genre"]）
