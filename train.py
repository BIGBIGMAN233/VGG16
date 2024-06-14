<<<<<<< HEAD
import torch
from torch.utils.tensorboard import SummaryWriter

torch.cuda.is_available()
import torchvision
from torch.utils.data import DataLoader
from module import *

# 准备数据
train_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=torchvision.transforms.ToTensor()
                                          , download=True)
test_data = torchvision.datasets.CIFAR10(root='data', train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)

# 加载数据
train_data_load = DataLoader(train_data, batch_size=64)
test_data_load = DataLoader(test_data, batch_size=64)

# 设置gpu
device = torch.device('cuda:1')
# 创建模型
vgg = VGG16()
vgg.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learn_rate = 0.01
# 优化器
optimizer = torch.optim.SGD(vgg.parameters(), lr=learn_rate)

# 参数设定
total_train_step = 0
train_epoch = 50
total_test_step = 0

# 添加tensorboard
writer = SummaryWriter('logs_train')
temp_accuracy = 0
# 开始训练
for i in range(train_epoch):

    vgg.train()
    print('---------第{}轮训练开始---------'.format(i + 1))
    for data in train_data_load:
        img, target = data
        img = img.to(device)
        target = target.to(device)
        outputs = vgg(img)
        loss = loss_fn(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数: {}，损失值：{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 开始测试
    vgg.eval()
    total_test_loss = 0
    total_train_step = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_load:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            outputs = vgg(img)
            loss = loss_fn(outputs, target)
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy
            total_test_loss = total_test_loss + loss.item()
    print("总体损失值：{}".format(total_test_loss))
    print("总体精度：{}".format(total_accuracy / len(test_data)))
    # 存取最佳模型

    torch.save(vgg, 'results/vgg{}--best.pth'.format(i))
    print('模型已保存')
writer.close()
=======
import torch
from torch.utils.tensorboard import SummaryWriter

torch.cuda.is_available()
import torchvision
from torch.utils.data import DataLoader
from module import *

# 准备数据
train_data = torchvision.datasets.CIFAR10(root='data', train=True, transform=torchvision.transforms.ToTensor()
                                          , download=True)
test_data = torchvision.datasets.CIFAR10(root='data', train=False, transform=torchvision.transforms.ToTensor()
                                         , download=True)

# 加载数据
train_data_load = DataLoader(train_data, batch_size=64)
test_data_load = DataLoader(test_data, batch_size=64)

# 设置gpu
device = torch.device('cuda:1')
# 创建模型
vgg = VGG16()
vgg.to(device)

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learn_rate = 0.01
# 优化器
optimizer = torch.optim.SGD(vgg.parameters(), lr=learn_rate)

# 参数设定
total_train_step = 0
train_epoch = 50
total_test_step = 0

# 添加tensorboard
writer = SummaryWriter('logs_train')
temp_accuracy = 0
# 开始训练
for i in range(train_epoch):

    vgg.train()
    print('---------第{}轮训练开始---------'.format(i + 1))
    for data in train_data_load:
        img, target = data
        img = img.to(device)
        target = target.to(device)
        outputs = vgg(img)
        loss = loss_fn(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数: {}，损失值：{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 开始测试
    vgg.eval()
    total_test_loss = 0
    total_train_step = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_load:
            img, target = data
            img = img.to(device)
            target = target.to(device)
            outputs = vgg(img)
            loss = loss_fn(outputs, target)
            accuracy = (outputs.argmax(1) == target).sum()
            total_accuracy = total_accuracy + accuracy
            total_test_loss = total_test_loss + loss.item()
    print("总体损失值：{}".format(total_test_loss))
    print("总体精度：{}".format(total_accuracy / len(test_data)))
    # 存取最佳模型

    torch.save(vgg, 'results/vgg{}--best.pth'.format(i))
    print('模型已保存')
writer.close()
>>>>>>> a341e64481184ad887d7b06f3f36ca64c9313329
