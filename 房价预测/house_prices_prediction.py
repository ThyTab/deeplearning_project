import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader,TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load and preprocess the data

def load_and_preprocess_data():
    # Load
    train_data = pd.read_csv('deeplearning_project/房价预测/house-prices-advanced-regression-techniques/train.csv')
    test_data = pd.read_csv('deeplearning_project/房价预测/house-prices-advanced-regression-techniques/test.csv')

    # Preprocess
    all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))   #dataframe数据类型
    #将连续数值的特征归一化,空数据位置替换为均值
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index   #all_features.dtypes是一个series数据
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))   #归一化
    all_features = all_features.fillna(0)   #fillna(0)将空数据位置替换为0
    #将离散数值的特征分为多个特征,并用数值表示
    all_features = pd.get_dummies(all_features,dtype=int,dummy_na=True)   #get_dummies用于类别特征独热编码

    # -> tensor
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(DEVICE)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(DEVICE)
    train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float32).view(-1, 1).to(DEVICE)
    train_labels = torch.log1p(train_labels)  #对标签取对数以减小数值差距

    return train_features, test_features, train_labels,test_data

# 2. train the model

class HousePriceModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = nn.Linear(train_features.shape[1],1)
        nn.init.normal_(self.output.weight, mean=0, std=0.01)
        nn.init.zeros_(self.output.bias)
    
    def forward(self,x):
        x = self.output(x)
        return x
    
def train_model(train_features,train_labels):
    model = HousePriceModel().to(DEVICE)
    Dataset = TensorDataset(train_features,train_labels)
    dataloader = DataLoader(Dataset,batch_size=128,shuffle=True)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))
    epochs = 500
    model.train()
    for epoch in range(epochs):
        for x_batch,y_batch in dataloader:
            pred_labels = model(x_batch)
            loss = criterion(pred_labels,y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')
    return model

# 3. predict and submit
def predict_and_submit(model,test_features,test_data):
    model.eval()
    with torch.no_grad():
        pred_test_labels_log10 = model(test_features)
        pred_test_labels = torch.exp(pred_test_labels_log10).cpu().numpy()
        test_data['SalePrice'] = pd.Series(pred_test_labels.reshape(1,-1)[0])
        submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
        submission.to_csv('deeplearning_project/房价预测/submission.csv',index=False)

# operate
if __name__ == '__main__':
    train_features, test_features, train_labels,test_data = load_and_preprocess_data()
    model = train_model(train_features,train_labels)
    predict_and_submit(model,test_features,test_data)