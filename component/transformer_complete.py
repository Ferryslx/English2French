"""
完整的 Transformer 模型架构实现
包含：输入模块、编码器、解码器、输出模块
参考论文："Attention Is All You Need" (Vaswani et al., 2017)
"""
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 工具函数
# ============================================================

def attention(query, key, value, mask=None, dropout=None):
    """
    缩放点积注意力机制
    :param query: 查询张量 [batch_size, seq_len, d_model]
    :param key:   键张量   [batch_size, seq_len, d_model]
    :param value: 值张量   [batch_size, seq_len, d_model]
    :param mask:  掩码张量，与 scores 形状匹配
    :param dropout: 随机失活层
    :return: (注意力输出, 注意力权重)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """克隆 N 个相同的模块，返回 ModuleList"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ============================================================
# 输入部分：词嵌入层 + 位置编码
# ============================================================

class Embedding(nn.Module):
    """词嵌入层（将 token 索引映射为词向量，并乘以 sqrt(d_model) 缩放）"""
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """位置编码层（使用正弦/余弦函数编码位置信息）"""
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000) / d_model)
        )
        position_value = position * div_term

        pe[:, 0::2] = torch.sin(position_value)
        pe[:, 1::2] = torch.cos(position_value)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================
# 编码器基本组件
# ============================================================

class LayerNorm(nn.Module):
    """层规范化"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.k = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_std = x.std(-1, keepdim=True)
        return self.k * (x - x_mean) / (x_std + self.eps) + self.b


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch = query.size(0)
        query, key, value = [
            model(x).view(batch, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, [query, key, value])
        ]

        x, attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        attn_x = x.transpose(1, 2).contiguous().view(batch, -1, self.head * self.d_k)
        return self.linears[-1](attn_x)


class FeedForward(nn.Module):
    """前馈全连接层（FFN）"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(F.relu(x))
        x = self.linear2(x)
        return x


class SublayerConnection(nn.Module):
    """
    子层连接结构（Pre-LN）
    顺序：层规范化 → 子层处理 → Dropout → 残差连接
    Pre-LN 训练更稳定，是 GPT/BERT/LLaMA 等现代 Transformer 的标准做法。
    """
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ============================================================
# 编码器
# ============================================================

class EncoderLayer(nn.Module):
    """
    编码器层
    流程：自注意力子层 → 残差连接 + LN → 前馈全连接子层 → 残差连接 + LN
    """
    def __init__(self, d_model, self_attn: MultiHeadAttention, feed_forward: FeedForward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.d_model = d_model
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask=None):
        # 第一子层：自注意力
        if mask is not None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        else:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        # 第二子层：前馈全连接
        x = self.sublayer[1](x, lambda x: self.feed_forward(x))
        return x


class Encoder(nn.Module):
    """编码器（由 N 个 EncoderLayer 堆叠 + 最后 LayerNorm）"""
    def __init__(self, layer: EncoderLayer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ============================================================
# 解码器
# ============================================================

class DecoderLayer(nn.Module):
    """
    解码器层（3 个子层）
    1. 掩码多头自注意力 + 残差连接 + LN
    2. 交叉注意力（编码器-解码器注意力）+ 残差连接 + LN
    3. 前馈全连接 + 残差连接 + LN
    """
    def __init__(self, d_model, self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention, feed_forward: FeedForward, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.d_model = d_model
        self.layers = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        # 第一子层：掩码自注意力（因果掩码，防止看到未来 token）
        x = self.layers[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 第二子层：交叉注意力（Q 来自解码器，K/V 来自编码器）
        x = self.layers[1](x, lambda x: self.src_attn(x, encoder_output, encoder_output, source_mask))
        # 第三子层：前馈全连接
        x = self.layers[2](x, lambda x: self.feed_forward(x))
        return x


class Decoder(nn.Module):
    """解码器（由 N 个 DecoderLayer 堆叠 + 最后 LayerNorm）"""
    def __init__(self, layer: DecoderLayer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)


# ============================================================
# 输出部分
# ============================================================

class Generator(nn.Module):
    """输出层：线性变换 + LogSoftmax，将解码器输出映射为词汇表概率分布"""
    def __init__(self, vocab_size, d_model):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


# ============================================================
# 完整 Transformer 模型
# ============================================================

class EncoderDecoder(nn.Module):
    """
    完整的 Transformer 架构
    流程：源序列嵌入 → 位置编码 → 编码器 → 解码器 → 输出层
    """
    def __init__(self,
                 source_embedding: Embedding,
                 source_position: PositionalEncoding,
                 encoder: Encoder,
                 target_embedding: Embedding,
                 target_position: PositionalEncoding,
                 decoder: Decoder,
                 generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.source_embedding = source_embedding
        self.source_position = source_position
        self.encoder = encoder
        self.target_embedding = target_embedding
        self.target_position = target_position
        self.decoder = decoder
        self.generator = generator

    def forward(self, source_x, target_y, source_mask=None, target_mask=None):
        """
        :param source_x:   编码器输入（原始 token 索引）
        :param target_y:   解码器输入（原始 token 索引）
        :param source_mask: 编码器掩码（padding mask）
        :param target_mask: 解码器掩码（causal mask，防止看到未来位置）
        :return: 目标词汇表的概率分布
        """
        embed_x = self.source_embedding(source_x)
        position_x = self.source_position(embed_x)
        result_x = self.encoder(position_x, source_mask)

        embed_y = self.target_embedding(target_y)
        position_y = self.target_position(embed_y)
        result_y = self.decoder(position_y, result_x, source_mask, target_mask)

        return self.generator(result_y)
