import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
from model import LeNet

# 超参数
epochs = 10
learn_rate = 0.001
batch_size = 16

save_model_path = "./model/LeNet.pth"

show_results = True


def main():
    # 预处理
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # 加载数据集
    train_set = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # 训练
    for epoch in range(epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        with tqdm(
            total=total_batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        ) as pbar:
            for i, data in enumerate(train_loader, 1):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 100 == 0 or i == total_batches:
                    average_loss = running_loss / i
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({"Loss": average_loss, "LR": lr})
                    pbar.update(100)

        epoch_loss = running_loss / total_batches
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} completed. Average Loss: {epoch_loss:.3f}, LR: {lr:.6f}"
        )

    print("Finished Training")

    # 保存模型
    torch.save(model.state_dict(), save_model_path)
    print("Model saved")

    # 测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy on the test images: %d %%" % (100 * correct / total))

    # 可视化结果
    if show_results:
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


if __name__ == "__main__":
    main()
