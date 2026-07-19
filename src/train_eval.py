import re
import os
import sys

import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.log import Logger

#设备选择
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#指定特殊的token
#起始标记
SOS_token =0
#结束标记
EOS_token =1
#最大句子长度不能超过10（包括标点）
MAX_LENGTH=10

#定义函数，进行文本处理
def normalize_string(s):
    #将字符串转成小写形式，并去除首位空白符号
    s=s.lower().strip()

    #在 .!?前加一个空格，用正则表达式的捕获组替换
    #参1：正则表达式，即要被替换的内容  参2：用来替换的内容  参3：要操作的字符串
    # 在常见英文标点（,.!?）前插入一个空格，用于分词前的文本规范化
    # ([.,!?]) : 捕获任意一个标点
    # r" \1"   : 在捕获的标点前加一个空格并保留原标点
    s=re.sub(r"([.,!?])", r" \1", s)

    #过滤非标准字符，保留大小写字母和基本标点符号，其他符号替换为空格
    # [^a-zA-Z.!?] : 非字母和非指定标点的字符(^就是取反，[^abc]就是除了abc都被匹配)
    # +            : 连续多个这类字符
    s=re.sub('[^a-zA-Z.!?]+',' ',s)
    return s

#数据预处理->加载数据到内存
def load_pairs():
    """加载并清洗双语句子对"""
    with open('../data/eng-fra-v2.txt','r',encoding='utf-8') as f:
        lines=f.readlines()
        my_pairs=[[normalize_string(s) for s in line.split('\t')]for line in lines]
    return my_pairs

def build_vocab(pairs):
    """从句子对列表中构建词汇表（只使用给定的pairs）"""
    english_word2index={'SOS':SOS_token,'EOS':EOS_token}
    french_word2index={'SOS':SOS_token,'EOS':EOS_token}

    for line in pairs:
        for word in line[0].split(' '):
            if word not in english_word2index:
                english_word2index[word]=len(english_word2index)
        for word in line[1].split(' '):
            if word not in french_word2index:
                french_word2index[word]=len(french_word2index)

    english_index2word={v:k for k,v in english_word2index.items()}
    french_index2word={v:k for k,v in french_word2index.items()}

    return english_word2index,english_index2word,len(english_word2index),french_word2index,french_index2word,len(french_word2index)


#数据预处理->构建Dataset对象
class MyPairsDataset(Dataset):
    def __init__(self,my_pairs,english_word2index,french_word2index):
        self.my_pairs=my_pairs
        self.english_word2index=english_word2index
        self.french_word2index=french_word2index
        self.sample_len=len(self.my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        index=max(0,min(index,self.sample_len-1))
        x=self.my_pairs[index][0]
        y=self.my_pairs[index][1]

        x=[self.english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        tensor_x=torch.tensor(x,dtype=torch.long,device=device)

        y=[self.french_word2index[word] for word in y.split(' ')]
        y.append(EOS_token)
        tensor_y=torch.tensor(y,dtype=torch.long,device=device)

        return tensor_x, tensor_y

#数据预处理->获取数据加载器
def get_dataloaders(train_pairs,val_pairs,english_word2index,french_word2index):
    train_dataset=MyPairsDataset(train_pairs,english_word2index,french_word2index)
    val_dataset=MyPairsDataset(val_pairs,english_word2index,french_word2index)

    train_dataloader=DataLoader(train_dataset,batch_size=1,shuffle=True)
    val_dataloader=DataLoader(val_dataset,batch_size=1,shuffle=False)

    return train_dataloader,val_dataloader

#模型构建（编码器，基于GRU）
class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size):
        """
        :param input_size: 编码器词嵌入层的输入维度
        :param hidden_size: 编码器的隐藏层维度，即隐藏层单元的个数
        """
        super(Encoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        #参1：词嵌入层的输入维度(即词汇表的大小)   参2：每个单词的特征维度
        self.embedding=nn.Embedding(input_size,hidden_size)
        #参1：输入的特征维度，即词嵌入维度（需要和词嵌入层的参2保持一致）  参2：隐藏层的维度   参3：批次维度是否是第一个维度，True的话格式就是[batch_size,seq_len,input_size]
        self.gru=nn.GRU(input_size=hidden_size,hidden_size=hidden_size,batch_first=True)

    def forward(self,input,hidden=None):
        """
        :param input: 输入的单词索引序列，即[batch_size,seq_len]->假如是[1,6]
        :param hidden:初始的隐藏状态,即[num_layers,batch_size,hidden_size]->[1,1,256]
        :return:
        """
        if hidden is None:
            hidden=torch.zeros(1,1,self.hidden_size,device=device)
        #输入[1，6]->[1，6，256]
        output=self.embedding(input)

        #输入：词嵌入层的输出，初始隐藏状态
        output,hidden=self.gru(output,hidden)

        #返回GRU输出和最终隐藏状态
        return output,hidden

#模型构建，解码器（无注意力机制）
class Decoder(nn.Module):
    def __init__(self,output_size,hidden_size):
        """
        :param output_size:解码器输出维度，即法语词汇表大小
        :param hidden_size:解码器隐藏层维度，即每个词向量的特征数（256）
        """
        super(Decoder,self).__init__()
        self.output_size=output_size
        self.hidden_size=hidden_size
        #创建词嵌入层
        self.embedding=nn.Embedding(output_size,hidden_size)
        #创建GRU层
        # 参1：输入的特征维度，即词嵌入维度（需要和词嵌入层的参2保持一致）  参2：隐藏层的维度   参3：批次维度是否是第一个维度，True的话格式就是[batch_size,seq_len,input_size]
        self.gru=nn.GRU(hidden_size,hidden_size,batch_first=True)
        #创建线性层
        self.out=nn.Linear(hidden_size,output_size)

    def forward(self,input,hidden=None):
        if hidden is None:
            hidden=torch.zeros(1,1,self.hidden_size,device=device)
        #input->假如[1,1]
        #output->[1,1,256]
        output=self.embedding(input)
        output=F.relu(output)
        #输入时：output->[batch_size,seq_len,hidden_size]     hidden->[num_layers,batch_size,hidden_size]
        #输出时：output格式同上
        output,hidden=self.gru(output,hidden)
        #线性层需要二维数据，output是三维的，需要改成output[0]
        output=self.out(output[0])

        return output,hidden

#模型构建，解码器（加入注意力机制）
class AttentionDecoder(nn.Module):
    def __init__(self,output_size,hidden_size,dropout_p=0.1,max_length=MAX_LENGTH):
        """
        :param output_size: 目标词汇表大小（法语）
        :param hidden_size:隐藏层维度（和编码器一致）
        :param dropout_p:随机失活概率
        :param max_length:句子最大长度（超过截断，不足补齐）
        """
        super(AttentionDecoder,self).__init__()
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.dropout_p=dropout_p
        self.max_length=max_length

        #创建词嵌入层
        #输入[1,1]->[1,1,256]
        self.embedding=nn.Embedding(output_size,hidden_size)

        #注意力权重计算层：计算查询向量和编码器输出的匹配程度
        #参1：拼接后的查询向量和隐藏状态[1,1,512]
        #参2：注意力权重分布->[1,1,10]最大10个词
        #Linear只能传入二维，后续需要[0]处理
        self.attn=nn.Linear(hidden_size*2,max_length)

        #注意力融合层，将词嵌入和注意力权重进行融合
        self.attn_combine=nn.Linear(hidden_size*2,hidden_size)

        #创建随机失活层
        self.dropout=nn.Dropout(p=dropout_p)

        #创建GRU层，处理序列数据，维持隐藏状态
        self.gru=nn.GRU(hidden_size,hidden_size,batch_first=True)

        #输出层
        #将GRU隐藏状态映射为法语词汇表大小
        self.output=nn.Linear(hidden_size,output_size)

    #input：当前时间步的输入词索引->[1,1]   encoder_outputs:编码器所有时间步的输出->[batch_size,seq_len,hidden]
    def forward(self,input,encoder_outputs,hidden=None):
        if hidden is None:
            hidden=torch.zeros(1,1,self.hidden_size,device=device)
        #词嵌入层[1,1]->[1,1,256]
        embed=self.embedding(input)
        embed=self.dropout(embed)

        #计算注意力权重
        attn_weights=F.softmax(self.attn(torch.cat((embed[0],hidden[0]),dim=1)),dim=-1)

        #计算注意力上下文
        attn_applied=torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        #注意力融合层
        output=torch.cat((embed[0],attn_applied[0]),dim=1)      #[1,512]
        output=self.attn_combine(output).unsqueeze(0)                   #[1,1,256]

        #激活函数
        output=F.relu(output)

        #GRU层
        output,hidden=self.gru(output,hidden)

        #输出层
        output=self.output(output[0])       #[1,4345]

        #参1：当前时间步的输出概率分布
        #参2：更新后的隐藏状态（本次的隐藏状态）
        #参3：注意力权重分布
        return output,hidden,attn_weights

lr,epochs,teacher_forcing_ratio,print_interval_num,plot_interval_num=1e-4,20,0.5,1000,100
"""
    :param lr: 学习率
    :param epochs: 训练轮数
    :param teacher_forcing_ratio:教师强制比例
    :param print_interval_num: 输出信息打印间隔
    :param plot_interval_num: 绘图间隔（每训练plot_interval_num条绘图一次）
    :return:
"""

#构建模型内部迭代训练函数->即：完成单批次的训练过程，完成一个样本的编码->解码->反向传播->优化参数...
def lone_batch_train(x,y,encoder,decoder,adam_encoder,adam_decoder,loss_function):
    """
    :param x: 输入序列，即英语句子，形状为[1，seq_len]
    :param y: 输出序列，即法语句子，形状为[1,seq_len]
    :param encoder: 编码器
    :param decoder: 解码器
    :param adam_encoder: 编码器优化器
    :param adam_decoder: 解码器优化器
    :param loss_function: 损失函数
    :return:
    """
    #编码阶段，将输入序列转换为上下文向量
    encoder_output,encoder_hidden=encoder(x)

    #解码器参数准备
    #限制编码器实际输出，固定句子长度为10
    encoder_output_c=torch.zeros(MAX_LENGTH,decoder.hidden_size)
    for idx in range(encoder_output.shape[1]):
        encoder_output_c[idx]=encoder_output[0,idx]
    #解码器初始隐藏状态
    decoder_hidden=encoder_hidden

    #解码器初始输出
    decoder_input=torch.tensor([[SOS_token]],device=device)

    #初始化损失值
    my_loss,y_len=0.0,y.shape[1]
    #根据概率值决定是否用教师强制(50%的概率)
    use_teacher_forcing=True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        #走这里，说明用teacher_forcing
        for i in range(y_len):
            #如果是填充了0，就把填充的mask掉
            if y[0][i].item() == EOS_token:
                break
            #解码器前向传播
            #decoder_input->[1,1],decoder_hidden->[1,1,256],encoder_output_c->[10,256]
            decoder_output,decoder_hidden,attn_weights=decoder(decoder_input,encoder_output_c,decoder_hidden)
            #获取当前时间步真实标签
            target_y=y[0][i].view(1)        #一维
            #累加损失
            my_loss+=loss_function(decoder_output,target_y)
            #下个时间步的输入直接使用真实标签
            decoder_input=y[0][i].view(1,-1)        #二维
    else:
        #非teacher forcing
        for i in range(y_len):
            # 解码器前向传播
            # decoder_input->[1,1],decoder_hidden->[1,1,256],encoder_output_c->[10,256]
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, encoder_output_c, decoder_hidden)
            # 获取当前时间步真实标签
            target_y = y[0][i].view(1)  # 一维
            # 累加损失
            my_loss += loss_function(decoder_output, target_y)
            # 获取预测的下一个词（即获取概率最高的词的索引）
            topv,topi=decoder_output.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            #到这，说明没有预测到结束标记，则：将预测的词作为下一步的输入
            decoder_input=topi.detach()

    #反向传播和参数更新
    encoder.zero_grad()
    decoder.zero_grad()

    my_loss.backward()
    adam_encoder.step()
    adam_decoder.step()

    #返回平均损失
    return my_loss.item()/y_len


#构建模型评估函数->在验证集上完成单个批次的前向传播和损失计算（无教师强制、无梯度更新）
def lone_batch_eval(x,y,encoder,decoder,loss_function):
    with torch.no_grad():
        encoder_output,encoder_hidden=encoder(x)

        encoder_output_c=torch.zeros(MAX_LENGTH,decoder.hidden_size,device=device)
        for idx in range(encoder_output.shape[1]):
            encoder_output_c[idx]=encoder_output[0,idx]
        decoder_hidden=encoder_hidden

        decoder_input=torch.tensor([[SOS_token]],device=device)

        my_loss,y_len=0.0,y.shape[1]
        for i in range(y_len):
            if y[0][i].item()==EOS_token:
                break
            decoder_output,decoder_hidden,attn_weights=decoder(decoder_input,encoder_output_c,decoder_hidden)
            target_y=y[0][i].view(1)
            my_loss+=loss_function(decoder_output,target_y)

            topv,topi=decoder_output.topk(1)
            if topi.squeeze().item()==EOS_token:
                break
            decoder_input=topi.detach()

    return my_loss.item()/y_len


#构建模型训练函数->即：完成所有批次的训练过程，即：多轮，多批次
def train_seq2seq():
    loss_records=[]
    logfile_name = 'English2French'
    logger=Logger('../',logfile_name).get_logger()
    logger.info("========== Seq2Seq 英译法训练开始 ==========")

    # 1. 加载数据
    my_pairs=load_pairs()
    logger.info(f"总样本数：{len(my_pairs)}")

    # 2. 在构建词表之前划分训练集和验证集（只用训练集构建词表）
    train_pairs,val_pairs=train_test_split(my_pairs,test_size=0.2,random_state=26)
    logger.info(f"训练集大小：{len(train_pairs)}，验证集大小：{len(val_pairs)}")

    # 3. 只用训练集构建词表
    english_word2index,english_index2word,english_word_num,french_word2index,french_index2word,french_word_num=build_vocab(train_pairs)
    logger.info(f"英文词表大小：{english_word_num}，法文词表大小：{french_word_num}")

    # 4. 获取数据加载器
    train_dataloader,val_dataloader=get_dataloaders(train_pairs,val_pairs,english_word2index,french_word2index)

    # 5. 初始化模型
    encoder=Encoder(english_word_num,256).to(device)
    decoder=AttentionDecoder(french_word_num,256,0.2,10).to(device)
    logger.info("编码器/解码器初始化完成")

    encoder_optimizer=optim.Adam(encoder.parameters(),lr=lr)
    decoder_optimizer=optim.Adam(decoder.parameters(),lr=lr)
    criterion=nn.CrossEntropyLoss()

    for epoch in range(1,epochs+1):
        logger.info(f"----- 第 {epoch}/{epochs} 轮开始 -----")

        # 模型训练
        print_loss_total,plot_loss_total=0.0,0.0
        start_time=time.time()

        for item,(x,y) in enumerate(tqdm(train_dataloader),start=1):
            my_loss=lone_batch_train(x,y,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)

            print_loss_total+=my_loss
            plot_loss_total+=my_loss

            if item%print_interval_num==0:
                print_loss_avg=print_loss_total/print_interval_num
                print_loss_total=0.0
                msg=f'轮次：{epoch}，训练平均损失:{print_loss_avg:.4f}，耗时：{time.time()-start_time:.4f}s'
                print(msg)
                logger.info(msg)

            if item%plot_interval_num==0:
                plot_loss_avg=plot_loss_total/plot_interval_num
                plot_loss_total=0.0
                loss_records.append({
                    "epoch":epoch,"step":item,"average_loss":plot_loss_avg,"type":"train"
                })

        # 一轮训练完毕，保存模型
        model_path_enc=f'../model/encoder_epoch{epoch}.pth'
        model_path_dec=f'../model/decoder_epoch{epoch}.pth'
        torch.save(encoder.state_dict(),model_path_enc)
        torch.save(decoder.state_dict(),model_path_dec)
        logger.info(f"模型已保存：{model_path_enc}，{model_path_dec}")

        # 验证集评估
        logger.info(f"轮次：{epoch}，开始在验证集上评估...")
        val_print_loss_total,val_plot_loss_total=0.0,0.0
        val_start_time=time.time()

        for item,(x,y) in enumerate(tqdm(val_dataloader),start=1):
            val_loss=lone_batch_eval(x,y,encoder,decoder,criterion)

            val_print_loss_total+=val_loss
            val_plot_loss_total+=val_loss

            if item%print_interval_num==0:
                val_print_loss_avg=val_print_loss_total/print_interval_num
                val_print_loss_total=0.0
                msg=f'轮次：{epoch}，验证平均损失:{val_print_loss_avg:.4f}，耗时：{time.time()-val_start_time:.4f}s'
                print(msg)
                logger.info(msg)

            if item%plot_interval_num==0:
                val_plot_loss_avg=val_plot_loss_total/plot_interval_num
                val_plot_loss_total=0.0
                loss_records.append({
                    "epoch":epoch,"step":item,"average_loss":val_plot_loss_avg,"type":"val"
                })

        logger.info(f"----- 第 {epoch}/{epochs} 轮结束 -----")

    # 保存损失到CSV
    df=pd.DataFrame(loss_records)
    df.to_csv("../loss/train_val_loss.csv",index=False)
    logger.info("训练和验证损失已保存到 loss/train_val_loss.csv")

    # 绘制损失曲线
    plt.figure()
    train_df=df[df["type"]=="train"]
    val_df=df[df["type"]=="val"]
    plt.plot(train_df["step"],train_df["average_loss"],label="Train Loss")
    plt.plot(val_df["step"],val_df["average_loss"],label="Val Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Seq2Seq Training and Validation Loss")
    plt.legend()
    plt.savefig('../figures/seq2seq_loss.png',dpi=300)
    plt.show()

    logger.info("========== Seq2Seq 英译法训练结束 ==========")


if __name__ == '__main__':
    train_seq2seq()