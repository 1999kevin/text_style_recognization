# ==================  提交 Notebook 训练模型结果数据处理参考示范  ==================
# 导入相关包
import copy
import os
import random
import numpy as np
import jieba as jb
import jieba.analyse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f
from torchtext import data
from torchtext import datasets
from torchtext.data import Field
from torchtext.data import Dataset
from torchtext.data import Iterator
from torchtext.data import Example
from torchtext.data import BucketIterator

# ------------------------------------------------------------------------------
# 本 cell 代码仅为 Notebook 训练模型结果进行平台测试代码示范
# 可以实现个人数据处理的方式，平台测试通过即可提交代码
#  -----------------------------------------------------------------------------

def load_data(path):
    """
    读取数据和标签
    :param path:数据集文件夹路径
    :return:返回读取的片段和对应的标签
    """
    sentences = [] # 片段
    target = [] # 作者

    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(path)
    for file in files:
        if not os.path.isdir(file) and not file[0] == '.':
            f = open(path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index,line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    return list(zip(sentences, target))

# 定义Field
TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x), lower=True, use_vocab=True)
LABEL = Field(sequential=False, use_vocab=False)
FIELDS = [('text', TEXT), ('category', LABEL)]

# 读取数据，是由tuple组成的列表形式
mydata = load_data(path='dataset/')

# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)
# 构建中文词汇表
TEXT.build_vocab(dataset)

# =========================  Notebook 训练模型网络结构参考示范  =========================
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.lstm = torch.nn.LSTM(1,64)
        self.fc1 = nn.Linear(64,128)
        self.fc2 = nn.Linear(128,5)

    def forward(self,x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        output,hidden = self.lstm(x.unsqueeze(2).float())
        h_n = hidden[1]
        out = self.fc2(self.fc1(h_n.view(h_n.shape[1],-1)))
        return out

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)
        # self.word_embeddings.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=2,bidirectional=True,dropout=0.5)
        # self.decoder1 = nn.Linear(128*2, 32)
        self.decoder = nn.Linear(128*2, 5)
    
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]
        final = lstm_out[-1]
        # y1 = self.decoder1(final)
        y = self.decoder(final)
        return y

# ----------------------------- 请加载您最满意的模型 -------------------------------
# 加载模型(请加载你认为的最佳模型)
# 加载模型,加载请注意 model_path 是相对路径, 与当前文件同级。
# 如果你的模型是在 results 文件夹下的 temp.pth 模型，则 model_path = 'results/temp.pth'

# 创建模型实例
model = LSTM()
model_path = "results/temp2.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))

# -------------------------请勿修改 predict 函数的输入和输出-------------------------
def predict(text):
    """
    :param text: 中文字符串
    :return: 字符串格式的作者名缩写
    """
     # ----------- 实现预测部分的代码，以下样例可代码自行删除，实现自己的处理方式 -----------
    labels = {0: 'LX', 1: 'MY', 2: 'QZS', 3: 'WXB', 4: 'ZAL'}
    # 自行实现构建词汇表、词向量等操作
    # 将句子做分词，然后使用词典将词语映射到他的编号
    text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text) ]
    # 转化为Torch接收的Tensor类型
    text2idx = torch.Tensor(text2idx).long()
    
    # 模型预测部分
    results = model(text2idx.view(-1,1))
    prediction = labels[torch.argmax(results,1).numpy()[0]]
    # --------------------------------------------------------------------------

    return prediction