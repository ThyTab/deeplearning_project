# 本次的隐藏状态 = tanh(上次的隐藏状态 * 权重矩阵 + 当前的输入 * 权重矩阵 + 偏置)
# 本次的输出 = 本次的隐藏状态 * 权重矩阵 + 偏置
#RNN；逐步处理词向量，生成每个时间步的隐藏状态
#全连接层：将隐藏状态转换为输出

import torch 
import torch.nn as nn

#创建RNN模型
# 词向量维度   隐藏状态维度   隐藏层层数
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)

# 定义变量，表示输入
# 句子长度   句子个数(batch_size)   词向量维度
x = torch.randn(5,32,128)

# 定义变量，表示上一时刻隐藏状态
# 隐藏层层数   句子个数(batch_size)   隐藏状态维度
h0 = torch.randn(1,32,256)

#调用RNN处理
output, h1 = rnn(x, h0)
print(f"output shape: {output.shape}")
print(f"h1 shape: {h1.shape}")