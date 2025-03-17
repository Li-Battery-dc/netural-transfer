from WikiartDataset import WikiArtDataset
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":

    dataset = WikiArtDataset(root_dir="wikiart",mode='test',transform=None)

    dataloader = DataLoader(dataset,batch_size=4,shuffle=True)

    for images,labels in dataloader:

        print(images)#输出图像数据
        print(labels)#输出标签
        break