import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import utils
import time
import os

from lossNet import Vgg16
from geneNet import geneNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def train_geneNet(style_img_path, save_model_name="geneNet.pth",
                batch_size=5, num_image=5000, alpha=1e-6, epochs=5):
    '''
    param: style_img_path: 风格图片路径
    param: save_model_name: 保存模型名称
    param: batch_size: 批次大小
    param: alpha: content_loss/style_loss表示权重
    param: epochs: 训练轮数
    para: num_image: 导入的训练图片数量
    '''
    lossNet = Vgg16(requires_grad=False).to(device)
    image_size = 256
    style_img = utils.load_image(style_img_path, size=image_size)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style_img = style_transform(style_img)
    style_img = style_img.repeat(batch_size, 1, 1, 1).to(device)
    # 使用vgg提取指定特征层
    features = lossNet(utils.normalize_batch(style_img))
    gram_style = [utils.gram_matrix(y) for y in features]

    # 导入训练数据
    from folder_data_handle import FolderDataset
    dataset = FolderDataset(
        root_dir="./data-COCO/COCO-img",
        image_size=image_size,
        num_image=num_image,
        mode="random",
    )
    # 导入时已经可以选择随机导入,不打开shuffle
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              shuffle=False, generator=torch.Generator(device=device))

    # 训练生成网络
    gene_net = geneNet().to(device)
    optimizer = torch.optim.Adam(gene_net.parameters(), lr=0.001)

    # 日志信息
    loss_history = []
    torch.autograd.set_detect_anomaly(True)
    for e in range(epochs):
        gene_net.train()
        num_trained_images = 0
        for batch_i, (images, _) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            gene_img = gene_net(images)

            assert images.size() == gene_img.size(), \
                "images.size() is {}, gene_img.size() is {}".format(images.size(), gene_img.size())

            images = utils.normalize_batch(images)
            gene_img = utils.normalize_batch(gene_img)
            gene_features = lossNet(gene_img)
            image_features = lossNet(images)

            content_loss = nn.MSELoss()(gene_features.relu2_2, image_features.relu2_2)

            style_loss = 0
            gram_gene = [utils.gram_matrix(y) for y in gene_features]
            for gm_s, gm_g in zip(gram_style, gram_gene):
                style_loss += nn.MSELoss()(gm_g, gm_s[:len(images), :, :])

            loss = alpha * content_loss + style_loss    
            loss.backward()
            optimizer.step()

            # 设置学习率衰减
            if num_trained_images > 0 and num_trained_images % 500 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9

            if batch_i % 50 == 0:
                mesg = "{} Epoch {}:  [{}/{}]  content_loss: {:.6f}  style_loss: {:.6f}  total_loss: {:.6f} lr: {:.5f}".format(
                    time.ctime(), e + 1, num_trained_images, len(train_loader.dataset),
                    content_loss.item(),
                    style_loss.item(),
                    loss.item(),
                    optimizer.param_groups[0]['lr']
                )
                print(mesg)
                loss_history.append(loss.item())

            num_trained_images = num_trained_images + len(images)
            
            check_point_interval = 500
            check_point_dir = "./check_point"
            if check_point_dir is not None and batch_i % check_point_interval == 0:
                gene_net.eval().cpu()
                checkpoint_name = "check_point_{}_batch_id_{}.pth".format(e+1, batch_i)
                checkpoint_path = os.path.join(check_point_dir, checkpoint_name)
                torch.save(gene_net.state_dict(), checkpoint_path)
                gene_net.to(device).train()
    
    # 保存日志
    import matplotlib.pyplot as plt  # 用于绘制和保存图表
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.savefig('./images/loss_log/loss_curve.png')

    # 保存模型
    gene_net.eval().cpu()
    save_model_dir = "./saved_model"
    save_model_path = os.path.join(save_model_dir, save_model_name)
    torch.save(gene_net.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)