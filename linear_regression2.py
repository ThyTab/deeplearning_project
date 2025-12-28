import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pyplot as plt                 #可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False      #防止中文乱码

#创建数据集
def create_dataset():
    n_samples = 100
    n_features = 1   #方便可视化
    noise = 1
    w_true = torch.tensor([4.0]).reshape(-1, 1)
    b_true = 5.0
    x = 10 * torch.rand(n_samples, n_features)
    y = x@w_true + b_true + noise * torch.randn(n_samples, 1)
    return x, y, w_true,b_true

def train(x,y):
    #1.创建数据集对象与数据加载器
    dataset = TensorDataset(x,y)
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    #2.创建模型
    model = nn.Linear(1,1)
    #3.创建损失函数与优化器
    criterion = nn.SmoothL1Loss()
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    #4.循环训练
    epochs,loss_list = 100,[]
    for epoch in range(epochs):
        total_loss,total_simple = 0,0
        for train_x,train_y in dataloader:
            #前向传播
            y_pred = model(train_x)
            #计算损失
            loss = criterion(y_pred,train_y.reshape(-1,1))
            total_loss += loss.item()*train_x.shape[0]
            total_simple += train_x.shape[0]
            #梯度清零 + 反向传播 + 优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #保存本轮损失
        loss_list.append(total_loss/total_simple)
        print(f'第{epoch+1}轮,损失:{total_loss/total_simple:.4f},权重:{model.weight.data},偏置:{model.bias.data}')
    return loss_list,model.weight,model.bias





if __name__ == '__main__':
    x,y,coef,bias_true = create_dataset()
    loss_list,weight,bias = train(x,y)

    #可视化初始数据
    plt.figure(1)
    plt.scatter(x.numpy(),y.numpy())   
    plt.title('初始数据分布')
    #损失曲线
    plt.figure(2)
    plt.plot(range(len(loss_list)),loss_list)
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('损失曲线')
    plt.grid()
    #预测结果与真实结果对比
    plt.figure(3)
    plt.plot(x.numpy(),(x@coef+bias_true).detach().numpy(),label='真实值')
    plt.plot(x.numpy(),(x@weight+bias).detach().numpy(),label='预测值')
    plt.legend()

    plt.show()