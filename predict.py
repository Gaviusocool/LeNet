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
        transforms.Resize((28, 28)),
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
def show_result(image_path):
    model.eval()
    figure = plt.figure(num="MNIST Sample Prediction", figsize=(6, 6))

    img = Image.open(image_path).convert("L")
    img_tensor = transform(img).unsqueeze(0)  # 添加一个维度作为 batch
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probs, 1)
        confidence = max_prob.item()

    plt.title(f"Predicted: {predicted.item()}, Confidence: {confidence:.3f}")
    plt.axis("off")
    plt.imshow(img, cmap="gray")  # 不需要使用 squeeze() 了

    plt.show()


def main(image_path):
    predicted_label = predict_image(image_path)
    print("Predicted label:", predicted_label)
    show_result(image_path)


if __name__ == "__main__":
    image_path = "./data/MNIST/test/9998.jpg"  # 图像路径
    main(image_path)
