import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 输入尺寸为224x224
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转为三通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 用于预训练的MobileNetV2
])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class MultiTaskMobileNetV2(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super(MultiTaskMobileNetV2, self).__init__()

        # 加载预训练的 MobileNetV2
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)

        # 共享的特征提取部分
        self.features = mobilenet_v2.features

        # 任务1的分类头（分类任务）
        self.classifier_task1 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task1)
        )

        # 任务2的回归头
        self.classifier_task2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes_task2)
        )

    def forward(self, x):
        # 共享的特征提取
        x = self.features(x)
        x = x.mean([2, 3])  # 全局平均池化

        # 任务1的输出
        task1_output = self.classifier_task1(x)

        # 任务2的输出
        task2_output = self.classifier_task2(x)

        return task1_output, task2_output

model = MultiTaskMobileNetV2(num_classes_task1=10, num_classes_task2=1)  # 任务1是10类分类，任务2是回归任务

# 4. 损失函数（任务1使用交叉熵，任务2使用均方误差）
criterion_task1 = nn.CrossEntropyLoss()  # 任务1 - 分类任务
criterion_task2 = nn.MSELoss()           # 任务2 - 回归任务（假设任务2为回归，目标值是数字的平方）

# 5. 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()

    running_loss_task1 = 0.0
    running_loss_task2 = 0.0

    for inputs, labels in train_loader:
        # 标签需要转换为任务2的回归目标（假设我们回归数字的平方）
        labels_task1 = labels
        labels_task2 = labels.float() ** 2  # 任务2是标签的平方作为回归目标

        # 前向传播
        outputs_task1, outputs_task2 = model(inputs)

        # 计算损失
        loss_task1 = criterion_task1(outputs_task1, labels_task1)
        loss_task2 = criterion_task2(outputs_task2.squeeze(), labels_task2)

        # 总损失
        total_loss = loss_task1 + loss_task2

        # 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss_task1 += loss_task1.item()
        running_loss_task2 += loss_task2.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Task1 Loss: {running_loss_task1/len(train_loader):.4f}, Task2 Loss: {running_loss_task2/len(train_loader):.4f}")

    # 每个 epoch 保存一次模型
    torch.save(model.state_dict(), f'multi_task_mobilenetv2_epoch_{epoch+1}.pth')
    print(f"Epoch {epoch+1} 模型参数已保存！")


# 7. 测试循环
model.eval()
correct = 0
total = 0
task2_predictions = []
task2_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        # 标签需要转换为任务2的回归目标
        labels_task1 = labels
        labels_task2 = labels.float() ** 2

        # 前向传播
        outputs_task1, outputs_task2 = model(inputs)

        # 任务1的分类准确度
        _, predicted = torch.max(outputs_task1, 1)
        total += labels_task1.size(0)
        correct += (predicted == labels_task1).sum().item()

        # 任务2的回归输出
        task2_predictions.extend(outputs_task2.squeeze().cpu().numpy())
        task2_labels.extend(labels_task2.cpu().numpy())

# 输出任务1的准确度
accuracy_task1 = 100 * correct / total
print(f"Task1 Classification Accuracy: {accuracy_task1:.2f}%")

# 输出任务2的回归误差
task2_predictions = torch.tensor(task2_predictions)
task2_labels = torch.tensor(task2_labels)
task2_mse = criterion_task2(task2_predictions, task2_labels)
print(f"Task2 Mean Squared Error: {task2_mse.item():.4f}")