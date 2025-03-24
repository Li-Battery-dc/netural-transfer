import sys
import os

# 将上级目录添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from torchvision import transforms
from Wikiart_dataset_include_classifier.models.DualPathResNet18_UNet_without_SEblock import DualPathResNet18_UNet
from PIL import Image

def classify(image_path):
    # 加载模型
    model= DualPathResNet18_UNet(num_classes=11)
    checkpoint = torch.load('Wikiart_dataset_include_classifier/train_models/checkpoint/DualPathResNet18-UNet_lr0.03_sgd_batchsize64_epochs50_weight_decay0.00015.pth', map_location=torch.device('cpu'),weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    # 预处理图像
    image = Image.open('image_path')
    transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])  # 与训练时相同的预处理
    input_tensor = transform(image).unsqueeze(0)  # 增加batch维度

    # 推理
    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()

    class_names = [
        "abstract_painting", "cityscape", "genre_painting", "illustration",
        "landscape", "nude_painting", "portrait", "religious_painting",
        "sketch_and_study", "still_life", "Unknown Genre"
    ]

    print(f'Predicted class: {class_names[predicted_class]}')
    return class_names[predicted_class]