import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False      #防止中文乱码

# 1.加载数据集
def load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
    test_dataset = datasets.MNIST(root='./data',train=False,transform=transform,download=True)
    train_iter = DataLoader(train_dataset,batch_size=256,shuffle=True)
    test_iter = DataLoader(test_dataset,batch_size=256,shuffle=False)
    return train_iter,test_iter

# 2.定义模型
class softmax_regression(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(28*28,10)
        #参数初始化
        nn.init.normal_(self.linear.weight,mean=0,std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self,x):
        x = self.linear(x.view(-1,28*28))   #不用加softmax层，CrossEntropyLoss自带
        return x
    
# 3.训练模型
def train(train_iter,test_iter):
    model = softmax_regression()
    criterion = nn.CrossEntropyLoss()   #自带softmax
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    #开始训练
    epochs,loss_list = 10,[]
    for epoch in range(epochs):
        total_loss,total_sample = 0,0
        for train_x,train_y in train_iter:
            #前向传播
            y_pred = model(train_x)
            #计算损失
            loss = criterion(y_pred,train_y)
            total_loss += loss.item()*train_x.shape[0]
            total_sample += train_x.shape[0]
            #梯度清零 + 反向传播 + 优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(total_loss/total_sample)
    return loss_list,model

# 4.模型评估
def evaluate(model,test_iter):
    model.eval()   #切换模型模式
    correct,total = 0,0
    with torch.no_grad():   #关闭梯度计算
        for test_x,test_y in test_iter:
            y_pred = model(test_x)
            y_ans = torch.argmax(y_pred,dim=1)   #取最大值索引
            correct += (y_ans==test_y).sum().item()   #只用.sum()得出的是一个一阶张量
            total += test_x.shape[0]
    accuracy = correct/total
    return accuracy

# 5.可视化
def visualize(loss_list):
    plt.plot(range(1,len(loss_list)+1),loss_list)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('Softmax回归模型训练损失值曲线')
    plt.show()

if __name__ == '__main__':
    train_iter,test_iter = load_data()
    loss_list,model = train(train_iter,test_iter)
    accuracy = evaluate(model,test_iter)
    print(f'在测试集上的准确率为:{accuracy*100:.2f}%')
    visualize(loss_list)