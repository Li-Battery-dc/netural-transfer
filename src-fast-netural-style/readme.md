# 快速神经风格迁移

## 项目结构

### main.py  

主程序入口
stylize parameter:

- content_img_pth：输入图片的路径
- model_pth: 加载模型的路径
- output_name: 输出图片的名称，现在指定路径为images/output/output_name.jpg

### train.py

训练生成网络，默认使用本地data_COCO中的图片，没有传到git上
parameter:

- style_img_path: 风格图片路径
- batch_size: 批次大小
- num_image: 训练时从数据集导入的图片数量
- alpha: content_loss/style_loss表示权重
- epochs: 训练轮数

### COCO_data_handle.py

 处理 COCO 数据集的工具，现在没有用到

### folder_data_handle.py

处理文件夹数据集的工具

### geneNet.py

生成网络模型

### lossNet.py

损失网络（目前基于 VGG16）

### utils.py

工具函数

### check_point/.pth

模型训练的中间检查点

### images/

图片文件夹（存储输入和输出图片）

### saved_model/.pth

保存的模型文件

