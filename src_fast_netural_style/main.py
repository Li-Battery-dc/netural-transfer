import time
import os

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np

from src_fast_netural_style.geneNet import geneNet
import src_fast_netural_style.utils as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def stylize(content_img_path, model_path, output_dir="src-fast-netural-style/images/output/", output_name="output.jpg"):
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
    
    utils.save_image(output_dir + output_name, output[0])

    end_time = time.time()
    print("Stylize image saved as", output_name)
    print("Time elapsed:", end_time - start_time)

def main():
    from train import train_geneNet
    style_img_dir = "src-fast-netural-style/images/style"
    batch_size = 5
    alpha = 2e-5
    epochs = 1
    num_image = 5000
    
    # for style_img_name in os.listdir(style_img_dir):
    #     style_img_path = os.path.join(style_img_dir, style_img_name)
    #     if os.path.isfile(style_img_path) and style_img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         # 动态生成模型名称
    #         model_name = f"{os.path.splitext(style_img_name)[0]}"
    #         print(f"Training with style image: {style_img_name}")
    #         # 调用 train_geneNet 进行训练
    #         train_geneNet(style_img_path=style_img_path, save_model_name=model_name, 
    #                       num_image=num_image, batch_size=batch_size, alpha=alpha, epochs=epochs)
    #         print(f"Model saved as: {model_name}")

    # for model_name in os.listdir("src-fast-netural-style/saved_model"):
    #     model_path = os.path.join("src-fast-netural-style/saved_model", model_name)
    #     if os.path.isfile(model_path) and model_name.lower().endswith('.pth'):
    #         content_img_path = "src-fast-netural-style/images/content/duck.jpg"
    #         output_dir = "src-fast-netural-style/images/output/duck/"
    #         output_name = f"{os.path.splitext(model_name)[0]}" + "_duck.jpg"
    #         stylize(content_img_path, model_path, output_dir=output_dir, output_name=output_name)

    style_img_path = style_img_dir + "/starry_night.jpg"
    train_geneNet(style_img_path=style_img_path, save_model_name="starry_night", 
                          num_image=num_image, batch_size=batch_size, alpha=alpha, epochs=epochs)
    for content in os.listdir("src-fast-netural-style/images/content"):
        content_img_path = os.path.join("src-fast-netural-style/images/content", content)
        if os.path.isfile(content_img_path) and content.lower().endswith(('.png', '.jpg', '.jpeg')):
            model_name = "starry_night.pth"
            model_path = os.path.join("src-fast-netural-style/saved_model", model_name)
            if os.path.isfile(model_path) and model_name.lower().endswith('.pth'):
                output_dir = "src-fast-netural-style/images/output/starrynight/"
                output_name = f"{os.path.splitext(model_name)[0]}" + "_" + content
                stylize(content_img_path, model_path, output_dir=output_dir, output_name=output_name)
    
    
    return

if __name__ == "__main__":
    main()