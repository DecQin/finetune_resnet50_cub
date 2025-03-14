import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm
from utils.weight_init import weight_init_kaiming
from utils.autoaugment import AutoAugment
from utils.ema import ModelEmaV2

# class ResNet50(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet50, self).__init__()
#         # 加载预训练的 ResNet50 模型
#         self.resnet50 = models.resnet50(pretrained=True)
#         self.resnet50.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # 替换原来ResNet模型的全连接层，将输出维度修改为与类别数量n_class一致，这里的512 * Config.expansion是根据ResNet架构的特征维度计算而来（Config.expansion可能是用于处理不同版本ResNet中通道数变化的系数）
#         self.resnet50.fc = nn.Linear(512 * 4, num_classes)
#         # 对新定义的全连接层应用kaiming初始化方法，有助于加快训练收敛速度并提升模型性能
#         self.resnet50.fc.apply(weight_init_kaiming)
#         # 替换最后的全连接层以适应特定的类别数
#         # self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

#     def forward(self, x):
#         N = x.size(0)
#         # 断言输入张量的形状是否符合预期，保证数据的格式正确
#         assert x.size() == (N, 3, 448, 448)
#         # 将输入数据传入基础模型（即选择的ResNet模型）进行前向计算
#         x = self.resnet50(x)
#         # 断言输出张量的形状是否符合预期，确保模型计算结果的维度正确
#         assert x.size() == (N, 200)
#         return x

#---------------------- 新增SE注意力模块 ----------------------
class SEBlock(nn.Module):
    """ Squeeze-and-Excitation 通道注意力模块 """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)      # [B, C]
        y = self.fc(y).view(b, c, 1, 1)      # [B, C, 1, 1]
        return x * y.expand_as(x)            # 通道加权

# ---------------------- 修改后的ResNet50模型 ----------------------
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        # 加载预训练ResNet50并移除最后两层
        self.base = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        
        # 添加SE注意力层（插入在base之后）
        self.se = SEBlock(channel=2048)  # ResNet50最后一层通道数为2048
        
        # 双线性池化层
        self.bilinear_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten()                   # [B, 2048]
        )
        self.pca = nn.Linear(2048**2, 8192)  # 添加在fc层前
        bilinear = self.pca(bilinear)
        # 分类头（处理双线性特征）
        self.fc = nn.Sequential(
            nn.Linear(2048**2, 1024),     # 降维处理
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)   # 最终分类层
        )
        self.fc.apply(weight_init_kaiming)  # 初始化

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)
        
        # 特征提取
        features = self.base(x)            # [B, 2048, 14, 14]
        
        # SE注意力加权
        features = self.se(features)       # [B, 2048, 14, 14]
        
        # 双线性池化
        feat_vec = self.bilinear_pool(features)  # [B, 2048]
        bilinear = torch.bmm(
            feat_vec.unsqueeze(2),          # [B, 2048, 1]
            feat_vec.unsqueeze(1)           # [B, 1, 2048]
        ).view(N, -1)                       # [B, 2048*2048]
        
        # 分类
        output = self.fc(bilinear)
        assert output.size() == (N, 200)
        return output
    

# 定义 LabelSmoothingLoss 类
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# CutMix数据增强
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    # 将 np.int 替换为 int
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
 
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, prob=0.3):  # 增加 prob 参数
    if torch.rand(1) < prob:  # 以 prob 的概率使用 CutMix
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam
    else:
        return x, y, y, 1  # 不使用 CutMix，返回原始数据

# 数据预处理和加载
train_transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),  # 随机旋转
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    AutoAugment(),  # autoaugment数据增强
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='/root/autodl-tmp/resnet_finetune_cub-master/dataset/train', transform=train_transform)
val_dataset = datasets.ImageFolder(root='/root/autodl-tmp/resnet_finetune_cub-master/dataset/test', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

# 加载模型
# model = models.resnet50(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 200)  # CUB-200-2011 有 200 个类别
model = ResNet50(num_classes=200)

# 定义优化器、损失函数和 EMA
optimizer = optim.SGD(model.parameters(), lr=0.01 , momentum=0.9, weight_decay=1e-4)
# scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
scheduler = CosineAnnealingLR(optimizer, T_max=360)
criterion = LabelSmoothingLoss(classes=200)
# ema = EMA(model, decay=0.999).to('cuda')
ema = ModelEmaV2(model, decay=0.9995).to('cuda')

# 训练函数
def train(model, train_loader, optimizer, criterion, ema, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(f"Type of data: {type(data)}, Type of target: {type(target)}")  # 添加调试信息
        data, target = data.to('cuda'), target.to('cuda')

        # r = np.random.rand(1)
        # if r < 0.3:
        #     data, targets = cutmix(data, target, alpha)
        #     output = model(data)
        #     target1, target2, lam = targets
        #     loss = lam * criterion(output, target1) + (1 - lam) * criterion(output, target2)
        # else:
        #     output = model(data)
        #     loss = criterion(output, target)

        # 使用CutMix
        data, target_a, target_b, lam = cutmix_data(data, target, alpha=0.4)

        output = model(data)
        loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)
        # loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

        train_loss += loss.item()
        _, predicted = output.max(1)
        # if r < 0.3:
        #     total += target1.size(0)
        #     correct += (lam * predicted.eq(target1).sum().item() + (1 - lam) * predicted.eq(target2).sum().item())
        # else:
        #     total += target.size(0)
        #     correct += predicted.eq(target).sum().item()
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    print(f'Epoch {epoch}: Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {100. * correct / total:.2f}%')
    return train_loss / len(train_loader), 100. * correct / total

# 验证函数
def validate(model, val_loader, criterion, ema):
    # model.eval()
    ema.module.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to('cuda'), target.to('cuda')
            # output = model(data)
            output = ema.module(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    print(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {100. * correct / total:.2f}%')
    return val_loss / len(val_loader), 100. * correct / total

# 训练和验证循环
num_epochs = 180
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

model = model.to('cuda')
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, ema, epoch)
    # ema.apply_shadow()
    val_loss, val_acc = validate(model, val_loader, criterion, ema)
    # ema.restore()
    scheduler.step()

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

# 可视化结果
plt.figure()

plt.plot(train_accuracies, color='b', label='Train Acc')
plt.plot(val_accuracies, color='r', label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend()

plt.savefig('result.png')
