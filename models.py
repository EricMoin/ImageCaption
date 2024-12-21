import torch
import torch.nn as nn
import torchvision
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, grid_size=7, num_layers=6, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # CNN特征提取器 (使用预训练的ResNet)
        self.cnn = torchvision.models.resnet50()
        # 移除最后的全连接层
        self.cnn = torch.nn.Sequential(*(list(self.cnn.children())[:-2]))
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, grid_size * grid_size, d_model))
        
        # 将CNN特征转换为Transformer期望的维度
        self.feature_projection = nn.Linear(2048, d_model)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 设置为True以匹配我们的输入格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # 通过CNN提取特征 [batch_size, 2048, 7, 7]
        x = self.cnn(x)
        
        # 重塑特征图 [batch_size, 2048, 49] -> [batch_size, 49, 2048]
        batch_size = x.size(0)
        x = x.reshape(batch_size, 2048, -1).permute(0, 2, 1)
        
        # 投影到d_model维度
        x = self.feature_projection(x)
        
        # 添加位置编码
        x = x + self.pos_embedding
        
        # 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 设置为True以匹配我们的输入格式
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        """生成方形的后续掩码"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, tgt, memory, tgt_mask=None):
        # tgt: [batch_size, seq_len]
        # memory: 编码器输出 [batch_size, 49, d_model]
        
        # 词嵌入和位置编码
        x = self.embedding(tgt)  # [batch_size, seq_len, d_model]
        x = self.pos_embedding(x.transpose(0, 1)).transpose(0, 1)  # 调整位置编码的维度顺序
        
        # 通过Transformer解码器
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        
        # 生成词表大小的输出分布
        output = self.output_layer(output)
        
        return output

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.encoder = TransformerEncoder(d_model=d_model)
        self.decoder = TransformerDecoder(vocab_size, d_model=d_model)
        
    def forward(self, img, tgt, tgt_mask=None):
        # 编码图像
        memory = self.encoder(img)
        
        # 解码生成描述
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        return output