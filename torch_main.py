###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
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

dataset = {}
path = "dataset/"
files= os.listdir(path)
for file in files:
    if not os.path.isdir(file) and not file[0] == '.': # 跳过隐藏文件和文件夹
        f = open(path+"/"+file, 'r',  encoding='UTF-8'); # 打开文件
        for line in f.readlines():
            dataset[line] = file[:-4]
name_zh = {'LX': '鲁迅', 'MY':'莫言' , 'QZS':'钱钟书' ,'WXB':'王小波' ,'ZAL':'张爱玲'} 

str_full = {}
str_full['LX'] = ""
str_full['MY'] = ""
str_full['QZS'] = ""
str_full['WXB'] = ""
str_full['ZAL'] = ""

for (k,v) in dataset.items():
    str_full[v] += k


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
        if not os.path.isdir(file):
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
mydata = load_data(path)

# 使用Example构建Dataset
examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), mydata))
dataset = Dataset(examples, fields=FIELDS)
# 构建中文词汇表
TEXT.build_vocab(dataset)
# 切分数据集
train, val = dataset.split(split_ratio=0.7)

# 生成可迭代的mini-batch
train_iter, val_iter = BucketIterator.splits(
    (train,val), # 数据集
    batch_sizes=(8,8),
    device=-1, # 如果使用gpu，此处将-1更换为GPU的编号
    sort_key=lambda x: len(x.text), 
    sort_within_batch=False,
    repeat=False
)
# Pytorch定义模型的方式之一：
# 继承 Module 类并实现其中的forward方法
# class Net(nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()                
#         self.lstm = torch.nn.LSTM(1,32)
#         self.fc1 = nn.Linear(32,128)
#         self.fc2 = nn.Linear(128,5)


#     def forward(self,x):
#         """
#         前向传播
#         :param x: 模型输入
#         :return: 模型输出
#         """
#         output,hidden = self.lstm(x.unsqueeze(2).float())
#         h_n = hidden[1]
#         out = self.fc2(self.fc1(h_n.view(h_n.shape[1],-1)))
#         return out


# Pytorch定义模型的方式之一：
# 继承 Module 类并实现其中的forward方法
class Net(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):
        
        super().__init__()          
        
        #embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm 层
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        
        #全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        #激活函数
        self.act = nn.Sigmoid()
        
    def forward(self, text, text_lengths):
        
        #text = [batch size,sent_length]
        embedded = self.embedding(text)
        #embedded = [batch size, sent_len, emb dim]
      
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # packed_output, (hidden, cell) = self.lstm(embedded)
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        
        #连接最后的正向和反向隐状态
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
                
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #激活
        outputs=self.act(dense_outputs)
        
        return outputs

    
    
# 创建模型实例
size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 5
num_layers = 2
bidirection = True
dropout = 0.2
model = Net(size_of_vocab, embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, 
                   bidirectional = True, dropout = dropout)



# pretrained_embeddings = TEXT.vocab.vectors
# model.embedding.weight.data.copy_(pretrained_embeddings)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


# train_acc_list, train_loss_list = [], []
# val_acc_list, val_loss_list = [], []
# for epoch in range(6):
#     train_acc, train_loss = 0, 0
#     val_acc, val_loss = 0, 0
#     for idx, batch in enumerate(train_iter):
#         text, text_lengths = batch.text
#         label =  batch.category
#         optimizer.zero_grad()
#         out = model(text, text_lengths)
#         loss = loss_fn(out,label.long())
#         loss.backward( retain_graph=True)
#         optimizer.step()
#         accracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
#         # 计算每个样本的acc和loss之和
#         train_acc += accracy*len(batch)
#         train_loss += loss.item()*len(batch)
        
#         print("\r opech:{} loss:{}, train_acc:{}".format(epoch,loss.item(),accracy),end=" ")
        
#定义度量
def binary_accuracy(preds, y):
    #四舍五入到最接近的整数
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    
    #初始化
    epoch_loss = 0
    epoch_acc = 0

    #设置为训练模式
    model.train()  
    
    for batch in iterator:
        #在每一个batch后设置0梯度
        optimizer.zero_grad()   
        text = batch.text  
        # text_lengths = 100 
        print(text.shape[0])
        #转换成一维张量
        lengths = torch.LongTensor(text.shape[0])
        predictions = model(text, lengths).squeeze()  
        #计算损失
        loss = criterion(predictions, batch.category)        
        #计算二分类精度
        acc = binary_accuracy(predictions, batch.category)   
        #反向传播损耗并计算梯度
        loss.backward()       
        #更新权重
        optimizer.step()      
        #损失和精度
        epoch_loss += loss.item()  
        epoch_acc += acc.item()     
    return epoch_loss / len(iterator), epoch_acc / len(iterator)        
    

def evaluate(model, iterator, criterion): 
    #初始化
    epoch_loss = 0
    epoch_acc = 0
    #停用dropout层
    model.eval()   
    #取消autograd
    with torch.no_grad():
        for batch in iterator:        
            text = batch.text           
            #转换为一维张量
            lengths = torch.LongTensor(text.shape[0])
            predictions = model(text, lengths).squeeze()           
            #计算损失和准确性
            loss = criterion(predictions, batch.category)
            acc = binary_accuracy(predictions, batch.category)            
            #跟踪损失和准确性
            epoch_loss += loss.item()
            epoch_acc += acc.item()        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


    # # 在验证集上预测
    # with torch.no_grad():
    #     for idx, batch in enumerate(val_iter):
    #         text, label = batch.text, batch.category
    #         out = model(text)
    #         loss = loss_fn(out,label.long())
    #         accracy = np.mean((torch.argmax(out,1)==label).cpu().numpy())
    #         # 计算一个batch内每个样本的acc和loss之和
    #         val_acc += accracy*len(batch)
    #         val_loss += loss.item()*len(batch)
            
    # train_acc /= len(train_iter.dataset)
    # train_loss /= len(train_iter.dataset)
    # val_acc /= len(val_iter.dataset)
    # val_loss /= len(val_iter.dataset)
    # train_acc_list.append(train_acc)
    # train_loss_list.append(train_loss)
    # val_acc_list.append(val_acc)
    # val_loss_list.append(val_loss)

N_EPOCHS = 5
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):     
    #训练模型
    train_loss, train_acc = train(model, train_iter, optimizer, criterion)    
    #评估模型
    valid_loss, valid_acc = evaluate(model, val_iter, criterion)    
    #保存最佳模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'results/temp.pth')
    
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')



# # 保存模型
# torch.save(model.state_dict(), 'results/temp.pth')

# 绘制曲线
# plt.figure(figsize=(15,5.5))
# plt.subplot(121)
# plt.plot(train_acc_list)
# plt.plot(val_acc_list)
# plt.title("acc")
# plt.subplot(122)
# plt.plot(train_loss_list)
# plt.plot(val_loss_list)
# plt.title("loss")
model = Net()
model_path = "results/temp.pth"
model.load_state_dict(torch.load(model_path))
print('模型加载完成...')
# 这是一个片段
text = "中国中流的家庭，教孩子大抵只有两种法。其一是任其跋扈，一点也不管，\
    骂人固可，打人亦无不可，在门内或门前是暴主，是霸王，但到外面便如失了网的蜘蛛一般，\
    立刻毫无能力。其二，是终日给以冷遇或呵斥，甚于打扑，使他畏葸退缩，彷佛一个奴才，\
    一个傀儡，然而父母却美其名曰“听话”，自以为是教育的成功，待到他们外面来，则如暂出樊笼的\
    小禽，他决不会飞鸣，也不会跳跃。"

labels = {0: '鲁迅', 1: '莫言', 2: '钱钟书', 3: '王小波', 4: '张爱玲'}

# 将句子做分词，然后使用词典将词语映射到他的编号
text2idx = [TEXT.vocab.stoi[i] for i in jb.lcut(text) ]

# 转化为Torch接收的Tensor类型
text2idx = torch.Tensor(text2idx).long()

# 预测
print(labels[torch.argmax(model(text2idx.view(-1,1)),1).numpy()[0]])

