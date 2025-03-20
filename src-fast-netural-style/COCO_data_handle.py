import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

from PIL import UnidentifiedImageError



class CustomCocoDetection(CocoDetection):
    def __getitem__(self, index):
        # 跳过损坏的或索引中不存在的图片
        try:
            return super().__getitem__(index)  # 调用父类方法加载数据
        except FileNotFoundError:
            # print(f"Warning: File not found for index {index}. Skipping...")
            return None  # 返回 None 表示跳过该图片
        except UnidentifiedImageError:
            # print(f"Warning: Unidentified image for index {index}. Skipping...")
            return None  # 返回 None 表示跳过该图片

# 使用自定义的 Cocoloader 类
class COCO_loader():
    def __init__(self, root, annFile, image_size, num_images=None):
        self.root = root
        self.annFile = annFile
        self.transform =  transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        self.dataset = CustomCocoDetection(root=root, annFile=annFile, transform=self.transform)
        self.valid_indices = [i for i in range(len(self.dataset)) if self.dataset[i] is not None]
        if num_images is not None:
            if len(self.valid_indices) < num_images:
                print(f"Warning: Only {len(self.valid_indices)} valid images found, less than the desired {num_images}.")
            else: 
                print(f"Using {num_images} images out of {len(self.valid_indices)} valid images.")
            self.valid_indices = self.valid_indices[:num_images]
    
    def load_data(self, batch_size):
        # 过滤掉 None 数据
        train_loader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            sampler=SubsetRandomSampler(self.valid_indices, generator=torch.Generator().manual_seed(42)),
        )
        return train_loader