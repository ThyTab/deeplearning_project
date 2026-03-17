import torch
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time


# 加载数据、分词、获取词表
def build_vocab():
    unique_words, all_words = [], []
    for line in open('./lyrics_generator/lyrics.txt', 'r', encoding='utf-8'):
        words = jieba.lcut(line)
        all_words.append(words)
        #去重
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    word_count = len(unique_words)
    # 构建词表
    word_to_index = {word:i for i, word in enumerate(unique_words)}
    # 歌词文本用词表索引表示
    corpus_index = []
    for words in all_words:
        for word in words:
            corpus_index.append(word_to_index[word])
        corpus_index.append(word_to_index[' '])
     
    return unique_words, word_to_index, word_count, corpus_index
    
# 数据预处理、构建数据集
class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_idx, num_chars):
        self.corpus_idx = corpus_idx
        self.num_chars = num_chars   #每个句子中词的数量
        self.word_count = len(self.corpus_idx)
        self.number = self.word_count//self.num_chars   #句子数量

    # 当使用len(obj)时，自动调用此方法
    def __len__(self):
        return self.number
    
    # 当使用obj[i]时，自动调用此方法,实现一个滑动窗口机制
    def __getitem__(self, idx):
        # idx:指的是词的索引
        # 确保start、end在合法范围内
        start = min(max(idx,0),self.word_count-self.num_chars-1)
        end = start + self.num_chars
        # 从文档中取出start-end之间的词
        x = self.corpus_idx[start:end]   # 输入
        y = self.corpus_idx[start+1:end+1]   # 输出

        return torch.tensor(x), torch.tensor(y)
    
# RNN模型
class TextGenerator(nn.Module):
    def __init__(self,unique_word_count):
        super().__init__()
        # 词嵌入层
        self.ebd = nn.Embedding(unique_word_count,128)
        # RNN   (词向量维度，隐藏层维度，RNN层数)
        self.rnn = nn.RNN(128,256,1)
        # 全连接层   (隐藏层维度，词表大小)
        self.out = nn.Linear(256,unique_word_count)

    def forward(self,inputs,hidden):
        # 词嵌入层处理
        embd = self.ebd(inputs)   #(batch,句子的长度，词向量维度)
        # RNN层处理   rnn传入的应该是(句子的长度，batch,词向量维度)
        output, hidden = self.rnn(embd.transpose(0,1),hidden)
        # 全连接层
        # 输入：(句子数量*batch,隐藏层维度)
        # 输出：(句子数量*batch,词表大小)
        output = self.out(output.reshape(-1,output.shape[-1]))

        return output, hidden

    # 初始化隐藏层
    def init_hidden(self,batch_size):
        return torch.zeros(1,batch_size,256)


# 训练模型

def train():
    unique_words, word_to_index,unique_word_count, corpus_index = build_vocab()
    lyrics = LyricsDataset(corpus_index,32)
    model = TextGenerator(unique_word_count)
    lyrics_dataloader = DataLoader(lyrics,batch_size=5,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    # 模型训练
    epochs = 100
    for epoch in range(epochs):
        start, iter_num, total_loss = time.time(), 0, 0.0
        for x,y in lyrics_dataloader:
            # 获取隐藏层初始值
            hidden = model.init_hidden(5)
            output, hidden = model(x, hidden)
            # y.shape(batch,seq_len,词向量维度)
            # output.shape(seq_len,batch,词向量维度)
            y = torch.transpose(y,0,1).reshape(shape=(-1,))   #先转换，再变为一维
            loss = criterion(output,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            iter_num += 1
        print(f"Epoch: {epoch+1}, Loss: {total_loss/iter_num:.4f}, Time: {time.time()-start:.2f}s")
    # 保存模型
    torch.save(model.state_dict(),'./lyrics_generator/model/text_generator.pth')


# 测试模型
def evaluate(start_word, sentence_length):
    unique_words, word_to_index,unique_word_count, corpus_index = build_vocab()
    model = TextGenerator(unique_word_count)
    model.load_state_dict(torch.load('./lyrics_generator/model/text_generator.pth'))
    hidden = model.init_hidden(1)
    word_idx = word_to_index[start_word]
    # 生成句子
    sentence = [start_word]
    for i in range(sentence_length):
        x = torch.tensor([[word_idx]])
        output, hidden = model(x, hidden)
        word_idx = torch.argmax(output,dim=1).item()
        sentence.append(unique_words[word_idx])
    for word in sentence:
        print(word,end='')


if __name__ == "__main__":
    # 1.获取数据、分词、获取词表
    unique_words, word_to_index,word_count, corpus_index = build_vocab()
    # 2.构建数据集
    dataset = LyricsDataset(corpus_index,num_chars=5)
    # 3.创建模型
    model = TextGenerator(word_count)
    # 4.训练并保存模型
    #train()
    # 5.测试模型
    evaluate('分手',20)

 