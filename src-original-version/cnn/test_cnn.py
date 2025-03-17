from define_cnn import MyStyleCNN
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyStyleCNN(num_classes=10)
    model.load_state_dict(torch.load("mystylecnn.pth"))
    model.to(device)
    model.eval()

    img_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=img_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"total: {total}, correct: {correct}")
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()