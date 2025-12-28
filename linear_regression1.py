import torch
from torch.utils.data import TensorDataset      #构建数据集对象
from torch.utils.data import DataLoader         #数据加载器
from torch import nn                         
from torch import optim                         #此模块里有优化器函数
from sklearn.datasets import make_regression    #创建线性回归模型数据集
import matplotlib.pyplot as plt                 #可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False      #防止中文乱码

'''
(1)准备训练集数据
(2)构建模型
(3)设置损失函数与优化器
(4)模型训练
'''
''' numpy对象 -> 张量tensor -> 数据集对象TensorDataset -> 数据加载器Dataloader '''
def creat_dataset():
    x,y,coef = make_regression(
        n_samples=100,   #样本数
        n_features=1,    #特征数
        noise=10,        #噪声
        coef=True,       #是否返回coef(权重,为一个数) 默认False
        bias=14.5,       #偏置
        random_state=23  #种子  
    )   # x,y为ndarray
    #转tensor
    x = torch.tensor(x,dtype=torch.float32)
    y = torch.tensor(y,dtype=torch.float32)
    return x,y,coef

def train(x,y,coef):
    #1.创建数据集对象   tensor -> 数据集对象 -> 数据加载器
    dataset = TensorDataset(x,y)
    #2.创建数据加载器对象
    #数据集对象,批次大小，是否打乱(训练集打乱,测试集不打乱)
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)
    #3.创建初始的线性模型
    #输入特征维度，输出特征维度
    model = nn.Linear(1,1)
    #4.创建损失函数对象
    criterion = nn.MSELoss()
    #5.创建优化器对象
    #模型参数,学习率
    optimizer = optim.SGD(model.parameters(),lr=0.01)   #SGD随机梯度下降
    #6.训练过程
    #6.1.定义变量:训练轮数,每轮的(平均)损失值,(本轮每批)训练总损失值,训练的样本数
    epochs,loss_list = 100,[]
    #6.2.开始训练,按轮训练
    for epoch in range(epochs):
        total_loss,total_sample = 0,0
        for train_x,train_y in dataloader:   #此步骤用于分批次训练
            #正向计算
            y_pred = model(train_x)
            #计算损失函数
            loss = criterion(y_pred,train_y.reshape(-1,1))
            total_loss += loss.item()*train_x.shape[0]   #分别是16,16,16,16,16,16,4
            total_sample += train_x.shape[0]
            #梯度清零 + 反向传播 + 梯度更新
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()   #梯度更新
        loss_list.append(total_loss/total_sample)
        print(f'轮数:{epoch+1},样本数:{total_sample},本轮的损失函数:{total_loss/total_sample}')
    print(f'最终结果   权重:{model.weight.data},偏置:{model.bias.data}')
    #可视化
    #绘制损失曲线
    plt.figure(1)
    plt.plot(range(epochs),loss_list)
    plt.title('损失值曲线变化图')
    plt.grid()   #绘制网格

    #绘制预测值与真实值的关系
    x=x.detach().numpy()
    y=y.detach().numpy()
    #样本点分布
    plt.figure(2)
    plt.scatter(x,y)
    #预测值与真实值折线图
    y_pred = model.weight.detach().numpy()*x+model.bias.detach().numpy()
    y_true = coef*x+14.5
    plt.plot(x,y_pred,color='red',label='预测值')
    plt.plot(x,y_true,color='blue',label='真实值')
    plt.grid()
    plt.legend()   #图例
    plt.show()


if __name__=='__main__':
    x,y,coef = creat_dataset()
    print(f'coef:{coef}')
    train(x,y,coef)

