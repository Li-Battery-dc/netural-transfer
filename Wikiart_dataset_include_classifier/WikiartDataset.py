import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import json

class WikiArtDataset(Dataset):
    def __init__(self,root_dir='wikiart',mode='train',transform=None):
        super().__init__()
        self.root_dir=root_dir
        self.mode = mode
        mode_file=os.path.join(self.root_dir,mode)#wikiart/train or wikiart/test or wikiart/val
        if mode=='train':
            #这里是对图像的预处理，可以再改
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.transform=transform
            
        elif mode=='test':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.transform=transform

        elif mode=='val':
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.transform=transform

        self.image_dir=os.path.join(mode_file,'images')
        self.label_dir=os.path.join(mode_file,'labels')

        self.folders = sorted(os.listdir(self.image_dir))

        self.image_paths=[]
        self.labels=[]

        for folder in self.folders:
            img_folder = os.path.join(self.image_dir,folder)
            label_file=os.path.join(self.label_dir,f"{folder}.json")

            with open(label_file,'r',encoding="utf-8") as f:
                label_data = json.load(f)

            for img_name,label in label_data.items():
                img_basename = os.path.basename(img_name)
                img_path=os.path.join(img_folder,img_basename)
                self.image_paths.append(img_path)
                self.labels.append(label)



    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        img_path = self.image_paths[idx]
        label=self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        #print(img_path)

        #如果有图像预处理，可以放在这里(可以修改)
        if self.transform:
            image = self.transform(image)

        return image,label

    
            