import random
import sys
import numpy as np
import yaml
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from Utils.ph_utils import PhDataset, PhDataset_test
from Models.PRNet import PhModel

# 读取配置
with open('/Config/train_config.yaml', 'r') as f:
    train_config = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 设置随机种子
def seed_everything(seed=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(32)

# 数据增强
train_transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize([112, 112]),
])
valid_transform = transforms.Compose([
    transforms.CenterCrop(384),
    transforms.Resize([112, 112]),
])

# 构建数据集
dataset_train = PhDataset(csv_path='train.csv', tfs=train_transform)
dataset_valid = PhDataset_test(csv_path='valid.csv', tfs=valid_transform)

Dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, pin_memory=True, num_workers=8)
Dataloader_valid = DataLoader(dataset_valid, batch_size=4, shuffle=False, drop_last=True, pin_memory=True,
                              num_workers=8)

# 初始化模型
model = PhModel().to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-5)


# 计算批次准确率
def binary_acc(preds, y):
    predicted_labels = torch.argmax(preds, dim=1)
    correct = (predicted_labels == y).sum()
    acc = correct.float() / y.size(0)
    return acc, correct


# 验证函数
def evaluate(model, criterion, dataloader):
    model.eval()
    class_correct = [0] * 2
    class_total = [0] * 2
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(dataloader, file=sys.stdout):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            correct = (preds == labels).cpu().numpy()
            for i in range(len(labels)):
                class_correct[labels[i].item()] += correct[i]
                class_total[labels[i].item()] += 1

    accuracy_per_class = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(2)]
    overall_accuracy = sum(class_correct) / sum(class_total) if sum(class_total) > 0 else 0
    avg_loss = total_loss / len(dataloader.dataset)
    return class_correct, class_total, accuracy_per_class, overall_accuracy, avg_loss


# 训练循环
if __name__ == '__main__':
    num_epochs = train_config['Train']['epoch']
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0

        for images, labels in tqdm(Dataloader_train, desc=f'Epoch [{epoch + 1}/{num_epochs}]'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc, correct = binary_acc(outputs, labels)
            train_loss += loss.item()
            train_correct += correct.item()

        class_correct, class_total, acc_per_class, overall_acc, val_loss = evaluate(model, criterion, Dataloader_valid)
        train_loss /= len(dataset_train)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_correct / len(dataset_train):.4f} | Val Acc: {overall_acc:.4f}")
        print(f"Val Class Acc: {acc_per_class}")


    torch.save(model, 'PRNet.pth')
