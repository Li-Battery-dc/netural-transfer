#用于批量生成样本数据
import pandas as pd
import json
import os
import glob

root ="wikiart"


parquet_path=root+"/raw_data"
train_path=root+"/train"
test_path=root+"/test"

images_path=root+"/images"
labels_path=root+"/labels"

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)
    
i=0
#读取 Parquet 文件
for file in sorted((os.listdir(parquet_path))):

    if i<5:
        images_savedpath=root+"/test"+"/images"
        labels_savedpath=root+"/test"+"/labels"
    else:
        images_savedpath=root+"/train"+"/images"
        labels_savedpath=root+"/train"+"/labels"


    if not os.path.exists(images_savedpath):
        os.makedirs(images_savedpath)
    if not os.path.exists(labels_savedpath):
        os.makedirs(labels_savedpath)

    prefix_to_remove = 'train-'
    if file.startswith(prefix_to_remove):
        # 去除前缀
        saved_filename = file[len(prefix_to_remove):]
    else:
        # 如果文件名不以指定前缀开头，则保持原样
        saved_filename = file

    readfile_path=parquet_path+"/"+file

    saved_filename= os.path.splitext(saved_filename)[0]  # 去除扩展名：'train''filename'
    df = pd.read_parquet(readfile_path, engine='pyarrow')
    # 初始化一个字典用于存储图像文件名和对应的 genre
    image_genre_mapping = {}

    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        # 提取图像字节数据
        image_data = row['image']['bytes']
        # 定义图像文件名
        image_savedpath = f'{images_savedpath}/{saved_filename}/image_{index}.jpg'
        image_name = f'{saved_filename}/image_{index}.jpg'
        # 将图像字节数据写入文件
        if not os.path.exists(f'{images_savedpath}/{saved_filename}'):
            os.makedirs(f'{images_savedpath}/{saved_filename}')

        with open(image_savedpath, 'wb') as img_file:
            img_file.write(image_data)
        # 将图像文件名和对应的 genre 添加到字典中
        image_genre_mapping[image_name] = row['genre']

    # 将 image_genre_mapping 字典保存为 JSON 文件
    with open(f'{labels_savedpath}/{saved_filename}.json', 'w') as json_file:
        json.dump(image_genre_mapping, json_file, indent=4)

    print("图像和 genre 信息已成功保存。")
    i+=1
