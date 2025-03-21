
from WikiartDataset import WikiArtDataset
from torch.utils.data import DataLoader
import torch

dataset_test = WikiArtDataset(root_dir="/scratch/stu1/netural-transfer/Wikiart_dataset/wikiart",mode='test',transform=None)
dataset_train = WikiArtDataset(root_dir="/scratch/stu1/netural-transfer/Wikiart_dataset/wikiart",mode='train',transform=None)
dataset_val = WikiArtDataset(root_dir="/scratch/stu1/netural-transfer/Wikiart_dataset/wikiart",mode='val',transform=None)
dataloader_test = DataLoader(dataset_test,batch_size=64,shuffle=True,num_workers=4,prefetch_factor=2)
dataloader_val = DataLoader(dataset_val,batch_size=64,shuffle=True,num_workers=4,prefetch_factor=2)
dataloader_train = DataLoader(dataset_train,batch_size=64,shuffle=True,num_workers=4,prefetch_factor=2)
batch_size = 64

if __name__ == "__main__":

    dataset_test = WikiArtDataset(root_dir="wikiart",mode='test',transform=None)
    dataset_train = WikiArtDataset(root_dir="wikiart",mode='train',transform=None)
    dataset_val = WikiArtDataset(root_dir="wikiart",mode='val',transform=None)
    dataloader_test = DataLoader(dataset_test,batch_size=256,shuffle=True)
    dataloader_val = DataLoader(dataset_val,batch_size=256,shuffle=True)
    dataloader_train = DataLoader(dataset_train,batch_size=256,shuffle=True)
    batch_size = 256

    for images,labels in dataloader_test:
        print(dataloader_test.dataset.mode)
        print(images.size())#输出图像数据
        print(labels)#输出标签
        break