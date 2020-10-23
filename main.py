# 导入相关包
import copy
import os
import numpy as np
import jieba as jb
import torch
import torch.nn as nn
import torch.nn.functional as f

from torchtext import data, datasets
from torchtext.data import Field,Dataset,Iterator,Example,BucketIterator

class Net(nn.Module):
    def __init__(self,vocab_size):
        super(Net,self).__init__()
        pass

    def forward(self,x):
        """
        前向传播
        :param x: 模型输入
        :return: 模型输出
        """
        pass

def processing_data(data_path, split_ratio = 0.7):
    """
    数据处理
    :data_path：数据集路径
    :validation_split：划分为验证集的比重
    :return：train_iter,val_iter,TEXT.vocab 训练集、验证集和词典
    """
    # --------------- 已经实现好数据的读取，返回和训练集、验证集，可以根据需要自行修改函数 ------------------
    sentences = [] # 片段
    target = [] # 作者
    
    # 定义lebel到数字的映射关系
    labels = {'LX': 0, 'MY': 1, 'QZS': 2, 'WXB': 3, 'ZAL': 4}

    files = os.listdir(data_path)
    for file in files:
        if not os.path.isdir(file):
            f = open(data_path + "/" + file, 'r', encoding='UTF-8');  # 打开文件
            for index,line in enumerate(f.readlines()):
                sentences.append(line)
                target.append(labels[file[:-4]])

    mydata = list(zip(sentences, target))

    TEXT  = Field(sequential=True, tokenize=lambda x: jb.lcut(x), 
                       lower=True, use_vocab=True)
    LABEL = Field(sequential=False, use_vocab=False)

    FIELDS = [('text', TEXT), ('category', LABEL)]

    examples = list(map(lambda x: Example.fromlist(list(x), fields=FIELDS), 
                                     mydata))

    dataset = Dataset(examples, fields=FIELDS)

    TEXT.build_vocab(dataset)
    
    train, val = dataset.split(split_ratio=split_ratio)
    
    # BucketIterator可以针对文本长度产生batch，有利于训练
    train_iter, val_iter = BucketIterator.splits(
        (train,val), # 数据集
        batch_sizes=(16, 16),
        device=-1, # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.text), 
        sort_within_batch=False,
        repeat=False # 
    )
    # --------------------------------------------------------------------------------------------
    return train_iter,val_iter,TEXT.vocab

def model(train_iter, val_iter, Text_vocab,save_model_path):
    """
    创建、训练和保存深度学习模型

    """
    # --------------------- 实现模型创建、训练和保存等部分的代码 ---------------------
    pass
    # 保存模型（请写好保存模型的路径及名称）
    
    # --------------------------------------------------------------------------------------------

def evaluate_mode(val_iter, save_model_path):
    """
    加载模型和评估模型
    可以实现，比如: 模型训练过程中的学习曲线，测试集数据的loss值、准确率及混淆矩阵等评价指标！
    主要步骤:
        1.加载模型(请填写你训练好的最佳模型),
        2.对自己训练的模型进行评估

    :param val_iter: 测试集
    :param save_model_path: 加载模型的路径和名称,请填写你认为最好的模型
    :return:
    """
    # ----------------------- 实现模型加载和评估等部分的代码 -----------------------
    pass
    
    # ---------------------------------------------------------------------------

def main():
    """
    深度学习模型训练流程,包含数据处理、创建模型、训练模型、模型保存、评价模型等。
    如果对训练出来的模型不满意,你可以通过调整模型的参数等方法重新训练模型,直至训练出你满意的模型。
    如果你对自己训练出来的模型非常满意,则可以提交作业!
    :return:
    """
    data_path = "./dataset"  # 数据集路径
    save_model_path = "results/model.pth"  # 保存模型路径和名称
    train_val_split = 0.7 #验证集比重

    # 获取数据、并进行预处理
    train_iter, val_iter,Text_vocab = processing_data(data_path, split_ratio = train_val_split)

    # 创建、训练和保存模型
    model(train_iter, val_iter, Text_vocab, save_model_path)

    # 评估模型
    evaluate_mode(val_iter, save_model_path)


if __name__ == '__main__':
    main()