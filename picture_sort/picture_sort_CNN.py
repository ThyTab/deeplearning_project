import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10   #6W张(32,32,3)图片
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False      #防止中文乱码
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.准备数据集
def create_dataset():
    # 1.训练集
    train_dataset = CIFAR10(root="./picture_sort/data",train=True,transform=ToTensor(),download=False)
    # 2.测试集
    test_dataset = CIFAR10(root="./picture_sort/data",train=False,transform=ToTensor(),download=False)
    # 3.返回数据集
    return train_dataset,test_dataset

# 2.搭建神经网络
class ImageModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,3,1,0)
        self.pool1 = nn.MaxPool2d(2,2,0)

        self.conv2 = nn.Conv2d(6,16,3,1,0)
        self.pool2 = nn.MaxPool2d(2,2,0)

        self.linear1 = nn.Linear(576,120)
        self.linear2 = nn.Linear(120,84)
        self.output = nn.Linear(84,10)

    def forward(self,x):
        # 第1层：卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x = self.pool1(torch.relu(self.conv1(x)))
        
        # 第2层：卷积层(加权求和) + 激励层(激活函数) + 池化层(降维)
        x = self.pool2(torch.relu(self.conv2(x)))

        x = x.reshape(x.size(0),-1)
        # 第3层：全连接层(加权求和) + 激励层(激活函数)
        x = torch.relu(self.linear1(x))

        # 第4层：全连接层(加权求和) + 激励层(激活函数)
        x = torch.relu(self.linear2(x))

        # 第5层：输出层
        return self.output(x)
    
def train(train_dataset):
    model = ImageModule().to(device)
    #准备步骤
    dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    #循环训练
    epochs = 50
    model.train()   #切换模型模式
    for epoch in range(epochs):
        total_loss, total_num, total_correct, start = 0.0, 0, 0, time.time()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            #五步
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #记录统计
            total_num += len(y)
            total_loss += loss.item()*len(y)
            total_correct += (torch.argmax(y_pred, dim=-1) == y).sum().item()
        print(f"epoch:{epoch+1}, loss:{total_loss/total_num:.5f},acc:{total_correct/total_num:.2f},time:{time.time()-start:.2f}s")
    #保存模型
    torch.save(model.state_dict(),"./picture_sort/model/model.pth")


def evaluate(test_dataset):
    dataloader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    model = ImageModule().to(device)
    model.load_state_dict(torch.load("./picture_sort/model/model.pth"))
    total_correct, total_num = 0, 0
    for x,y in dataloader:
        model.eval()   #切换模型模式
        x,y = x.to(device),y.to(device)
        y_pred = model(x)
        y_pred = torch.argmax(y_pred,dim=-1)
        total_correct += (y_pred == y).sum().item()
        total_num += len(y)
    print(f"test acc:{total_correct/total_num:.2f}")

if __name__ == "__main__":

    train_dataset,test_dataset = create_dataset()
    #查看模型参数
    #summary(model,(3,32,32),batch_size=BATCH_SIZE)
    #train(train_dataset)
    evaluate(test_dataset)