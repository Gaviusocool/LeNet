import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from model import LeNet

model_path = "./model/LeNet.pth"

model = LeNet()
model.load_state_dict(torch.load(model_path))
model.eval()

# 预处理
transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
test_set = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)


# 预测
def predict_image(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0)  # 增加一个维度作为batch
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


# show result
def show_result():
    model.eval()
    figure = plt.figure(num="MNIST Sample Prediction", figsize=(15, 15))
    rows, cols = 5, 2  # 每行显示两个类别的样本

    for class_label in range(10):  # MNIST 数据集有 10 个类别
        # 寻找测试集中具有特定类别的样本
        class_samples = [
            i for i, (img, label) in enumerate(test_set) if label == class_label
        ]
        # 随机选择一个具有特定类别的样本
        sample_idx = class_samples[0]
        img, label = test_set[sample_idx]

        with torch.no_grad():
            outputs = model(img.unsqueeze(0))
            probs = torch.nn.functional.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probs, 1)
            confidence = max_prob.item()

        # 计算当前子图的索引
        idx = class_label + 1
        ax = figure.add_subplot(rows, cols, idx)
        ax.set_title(
            f"pred: {predicted.item()}, conf: {confidence:.3f}, label: {label}"
        )
        ax.axis("off")
        ax.imshow(img.squeeze(), cmap="gray")

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


def main(image_path):
    predicted_label = predict_image(image_path)
    print("Predicted label:", predicted_label)
    show_result()


if __name__ == "__main__":
    image_path = "./data/MNIST/raw/test/1.jpg"  # 图像路径
    main(image_path)
