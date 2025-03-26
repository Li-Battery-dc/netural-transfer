import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FolderDataset(Dataset):
    def __init__(self, root_dir, image_size, num_image=None, mode="random", seed=42):
        """
        Args:
            root_dir (str): 图片文件夹路径
            image_size (int): 调整图片大小并中心裁剪到该尺寸
            max_image_count (int, optional): 限制加载的图片数量（默认加载全部）
            mode (str): 选择模式，"random"（随机）或 "sequential"（顺序）
            seed (int): 随机种子（仅在 mode="random" 时生效）
        """
        self.root_dir = root_dir
        self.max_image_count = num_image
        self.mode = mode
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        self.image_paths = [
            entry.path 
            for entry in os.scandir(root_dir) 
            if entry.name.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        self._filter_paths()
    
    def _filter_paths(self):
        if self.max_image_count is not None and self.max_image_count > 0:
            if self.mode == "random":
                random.seed(self.seed)
                self.image_paths = random.sample(self.image_paths, min(self.max_image_count, len(self.image_paths)))
            elif self.mode == "sequential":
                self.image_paths = self.image_paths[:self.max_image_count]
            else:
                raise ValueError(f"Unsupported mode: {self.mode}. Choose 'random' or 'sequential'.")
            self.image_paths = self.image_paths[:self.max_image_count]
        print(f"load {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # 转为 RGB 格式
        image = self.transform(image)
        label = 0 # 支持后续拓展
        
        return image, label  # 无需返回标签


# def tets_folder_dataset():
#     # 加载数据集
#     dataset = FolderDataset(
#         root_dir="./data-COCO/COCO-img",
#         image_size=256,
#         num_image=10,
#         mode="random"
#     )

#     # 创建 DataLoader（返回的 labels 是空，但实际不需要）
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

#     # 测试数据加载
#     for batch_i, (image, _) in enumerate(dataloader):
#         print(batch_i, image.shape)  # 输出形状应为 [batch_size, 3, 256, 256]

# tets_folder_dataset()