import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import lr_scheduler

# 加载训练和测试数据集
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)


# 定义网络架构
class Net(nn.Module):
    # 初始化网络层
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 检查GPU是否可用，并将模型移到GPU上
if torch.cuda.is_available():
    print('GPU is available')
    print(f'Current GPU device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
else:
    print('GPU is not available, switching to CPU')

net = Net()
device = torch.device("cuda")
net.to(device)

# 定义优化器和学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 开始训练
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 0:
            print('Epoch: %d, step: %d, loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    scheduler.step()
    net.eval()
    correct_valid = 0
    total_valid = 0
    with torch.no_grad():
        for data in valid_data_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted == labels).sum().item()
    valid_accuracy = 100 * correct_valid / total_valid
    print('Epoch: %d, Validation accuracy: %.2f %%' % (epoch + 1, valid_accuracy))

# 开始测试
correct = 0
total = 0
with torch.no_grad():
    for data in test_data_loader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
