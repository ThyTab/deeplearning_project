import torch
import jieba   #jieba分词库
import torch.nn as nn

def dm01():
    text = "Faker无需向众神祈祷，因为众神传颂的是他的名号！"
    #分词
    words = jieba.lcut(text)
    print(words)
    #创建词嵌入层
    embed = nn.Embedding(len(words), 10)   #词表大小为len(words)，每个词向量维度为10
    #将文本转换为索引
    for i, word in enumerate(words):
        word_vector = embed(torch.tensor(i))   #随机的，每次都不一样
        print(word, word_vector)

if __name__ == "__main__":
    dm01()