#用于批量生成样本数据
import pandas as pd
import json
import os
import glob

root ="/Users/13102/Desktop/Introduction_AI/wikiart"

parquet_path=root+"/data"
images_path=root+"/images"
labels_path=root+"/labels"

#读取 Parquet 文件
for file in sorted((os.listdir(parquet_path))):

    file_name=parquet_path+"/"+file

    basename=os.path.basename(file_name)
    name_without_extension = os.path.splitext(basename)[0]  # 去除扩展名：'filename'
    df = pd.read_parquet(file_name, engine='pyarrow')
    # 初始化一个字典用于存储图像文件名和对应的 genre
    image_genre_mapping = {}

    # 遍历 DataFrame 的每一行
    for index, row in df.iterrows():
        # 提取图像字节数据
        image_data = row['image']['bytes']
        # 定义图像文件名
        if not os.path.exists(f'{images_path}/{name_without_extension}'):
            os.makedirs(f'{images_path}/{name_without_extension}')
        image_filename=f'{images_path}/{name_without_extension}/image_{index}.jpg'
        image_name = f'{name_without_extension}/image_{index}.jpg'
        # 将图像字节数据写入文件
        with open(image_filename, 'wb') as img_file:
            img_file.write(image_data)
        # 将图像文件名和对应的 genre 添加到字典中
        image_genre_mapping[image_name] = row['genre']

    # 将 image_genre_mapping 字典保存为 JSON 文件
    with open(f'{labels_path}/{name_without_extension}.json', 'w') as json_file:
        json.dump(image_genre_mapping, json_file, indent=4)

    print("图像和 genre 信息已成功保存。")
