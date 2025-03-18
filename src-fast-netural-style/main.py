import time
import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from vgg import Vgg16
from geneNet import geneNet
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def train_geneNet(style_img_path, dataset):
    vgg = Vgg16(requires_grad=False).to(device)
    style_img = utils.load_image(style_img_path).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style_img = style_transform(style_img)
    # 指定batch_size拓展
    batch_size = 4
    style_img = style_img.repeat(batch_size, 1, 1, 1).to(device)
    # 使用vgg提取指定特征层
    features = vgg(utils.normalize_batch(style_img))
    gram_style = [utils.gram_matrix(y) for y in features]

    # 导入训练数据
    image_size = 256
    data_transfrom = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(dataset, transform=data_transfrom)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    # 训练生成网络
    gene_net = geneNet().to(device)
    optimizer = torch.optim.Adam(gene_net.parameters(), lr=0.001)

    epochs = 10
    # 表示内容损失和风格损失的比例
    alpha = 1e-5
    for e in range(epochs):
        gene_net.train()
        count = 0
        for batch_i, (images, _) in enumerate(train_loader):
            num_batch = len(images)
            count += num_batch
            optimizer.zero_grad()

            images = images.to(device)
            gene_img = gene_net(images)

            images = utils.normalize_batch(images)
            gene_img = utils.normalize_batch(gene_img)
            gene_features = vgg(gene_img)
            image_features = vgg(images)

            content_loss = nn.MSELoss()(gene_features.relu2_2, image_features.relu2_2)

            style_loss = 0
            gram_gene = [utils.gram_matrix(y) for y in gene_features]
            for gm_s, gm_g in zip(gram_style, gram_gene):
                style_loss += nn.MSELoss()(gm_g, gm_s[:num_batch, :, :])

            loss = alpha * content_loss + style_loss    
            loss.backward()
            optimizer.step()
            if batch_i % 10 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent_loss: {:.6f}\tstyle_loss: {:.6f}\ttotal_loss: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_loader.dataset),
                                  content_loss.item(),
                                  style_loss.item(),
                                  loss.item()
                )
                print(mesg)
            
            check_point_interval = 500
            check_point_dir = "C:/Users/Lenovo/Desktop/人智导作业/期中大作业/src-fast-netural-style/check_point"
            if check_point_dir is not None and batch_i % check_point_interval == 0:
                gene_net.eval().cpu()
                checkpoint_name = "check_point_{}_batch_id_{}.pth".format(e, batch_i)
                checkpoint_path = os.path.join(check_point_dir, checkpoint_name)
                torch.save(gene_net.state_dict(), checkpoint_path)
                gene_net.to(device).train()
            
    # 保存模型
    gene_net.eval().cpu()
    save_model_dir = "C:/Users/Lenovo/Desktop/人智导作业/期中大作业/src-fast-netural-style/saved_model"
    save_model_name = "geneNet.pth"
    save_model_path = os.path.join(save_model_dir, save_model_name)
    torch.save(gene_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)

def stylize(content_img_path, model_path):
    content_img = utils.load_image(content_img_path).to(device)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_img = content_transform(content_img).unsqueeze(0).to(device)

    with torch.no_grad():
        gene_net = geneNet().to(device)
        gene_net.load_state_dict(torch.load(model_path))
        gene_net.eval()
        output = gene_net(content_img).cpu()
    
    output_name = "output_01.jpg"
    utils.save_image("C:/Users/Lenovo/Desktop/人智导作业/期中大作业/src-fast-netural-style/images/output" + output_name, output[0])

def main():
    '''
    param: style_img_path: 风格图片路径
    param: dataset: 训练数据集路径
    param: content_img_path: 内容图片路径
    param: model_path: 模型路径
    '''
    dataset = 
    train_geneNet("C:/Users/Lenovo/Desktop/人智导作业/期中大作业/src-fast-netural-style/images/style/picasso.jpg")
    # stylize("C:/Users/Lenovo/Desktop/人智导作业/期中大作业/src-fast-netural-style/images/content/flower.jpg", 
    #         "C:/Users/Lenovo/Desktop/人智导作业/期中大作业/src-fast-netural-style/saved_model/geneNet.pth")

if __name__ == "__main__":
    main()