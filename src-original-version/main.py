import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

from PIL import Image

img_size = 512 if torch.cuda.is_available() else 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# 数据预处理
preprocess = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = preprocess(img).unsqueeze(0)  # 添加batch维度
    return img.to(device, torch.float32)

# 输入图片
content_img = load_image("images/myPicture.jpg")
style_img = load_image("images/picasso.jpg")

input_img = content_img.clone()
# 使用随机图片初始化
# input_img = torch.rand_like(content_img)
#input_img = torch.zeros_like(content_img) 

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

# 加载预训练模型评估模式
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

# 计算loss的卷积层
content_layers_default = ['conv_5']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

from module import Normalization, ContentLoss, StyleLoss

def get_model_and_loss(cnn, style_img, content_img, 
                       content_layers=content_layers_default, 
                       style_layers=style_layers_default):
    
    # pytorch的标准化参数
    normalization = Normalization([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    model = model[:(i + 1)]
    return model, style_losses, content_losses

# 加权计算全损失函数
style_weight = 1000000
content_weight = 1

# 梯度下降
def run(cnn, content_img, style_img, input_img, num_steps=300):

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_model_and_loss(cnn, style_img, content_img)
    
    # 让 input_img 计算梯度，以便通过梯度下降优化输入。
    # 将模型设置为评估模式，确保推理时的行为正确。
    # 关闭模型参数的梯度计算，以节省计算资源。
    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = optim.LBFGS([input_img])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_score * style_weight + content_weight * content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                # 学习率衰减
                optimizer.param_groups[0]['lr'] *= 0.8
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return loss

        optimizer.step(closure)

    with torch.no_grad(): 
        input_img.data.clamp_(0, 1)

    print("Now the model look like:")
    print(model)
    return input_img

# 调用时使用input_img.clone()，避免修改原始数据
output_image = run(cnn, content_img, style_img, input_img.clone())

from show_picture import imshow
from matplotlib import pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
imshow(content_img, title='Content Image')
plt.subplot(2, 2, 2)
imshow(style_img, title='Style Image')
plt.subplot(2, 2, 3)
imshow(input_img, title='input Image')
plt.subplot(2, 2, 4)
imshow(output_image, title='Output Image')
plt.ioff()
plt.show()

