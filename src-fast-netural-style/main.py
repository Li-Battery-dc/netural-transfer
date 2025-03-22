import time
import os

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from geneNet import geneNet
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def stylize(content_img_path, model_path, output_name="output.jpg"):
    start_time = time.time()
 
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_img = utils.load_image(content_img_path)
    content_img = content_transform(content_img).unsqueeze(0).to(device)

    with torch.no_grad():
        gene_net = geneNet().to(device)
        gene_net.load_state_dict(torch.load(model_path))
        gene_net.eval()
        output = gene_net(content_img).cpu()
    
    utils.save_image("./images/output/" + output_name, output[0])

    end_time = time.time()
    print("Stylize image saved as", output_name)
    print("Time elapsed:", end_time - start_time)

def main():
    from train import train_geneNet
    style_img_path = "./images/style/monet.jpg"
    batch_size = 5
    alpha = 5e-6
    epochs = 2
    train_geneNet(style_img_path=style_img_path, save_model_name="monet.pth",num_image=6000,
                  batch_size=batch_size,alpha=alpha,epochs=epochs)
    # stylize(content_img_path="./images/content/yulan.jpg", 
    #         model_path="./check_point/check_point_2_batch_id_500.pth", 
    #         output_name="yulan_monet.jpg")
    return
    # print("stylize or train geneNet?")
    # print("1. stylize")
    # print("2. train geneNet")
    # choice = int(input())
    # if choice == 1:
    #     content_img_path = input("Content image path: ")
    #     model_path = input("Model path: ")
    #     output_name = input("Output name: ")
    #     stylize(content_img_path=content_img_path,
    #             model_path=model_path,
    #             output_name=output_name)
    # else:
    #     from train import train_geneNet
    #     style_img_path = input("Style image path for train: ")
    #     batch_size = int(input("Batch size: "))
    #     alpha = float(input("content_loss/style_loss rate: "))
    #     epochs = int(input("Epochs: "))
    #     train_geneNet(style_img_path=style_img_path,
    #                   batch_size=batch_size,
    #                   alpha=alpha,
    #                   epochs=epochs)
    #     return

if __name__ == "__main__":
    main()