import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, nhead=8, d_model=512, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        # 图像特征提取
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # 冻结backbone参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 特征投影层
        self.feature_projection = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # Transformer编码器和解码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """生成注意力掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt, tgt_mask=None):
        """
        src: 输入图像 [batch_size, channels, height, width]
        tgt: 目标序列 [batch_size, seq_len]
        tgt_mask: 目标序列的注意力掩码
        """
        # 提取图像特征
        features = self.backbone(src)  # [batch_size, 2048, 7, 7]
        
        # 投影到d_model维度
        features = self.feature_projection(features)  # [batch_size, d_model, 7, 7]
        
        # 重塑为序列
        batch_size = features.size(0)
        features = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)  # [batch_size, 49, d_model]
        
        # Transformer编码
        memory = self.transformer_encoder(features)  # [batch_size, 49, d_model]
        
        # 词嵌入
        tgt = self.embedding(tgt)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        tgt = tgt.transpose(0, 1)  # [seq_len, batch_size, d_model]
        tgt = self.pos_encoder(tgt)
        tgt = tgt.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer解码
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask
        )  # [batch_size, seq_len, d_model]
        
        # 生成词概率
        output = self.output_layer(output)  # [batch_size, seq_len, vocab_size]
        
        return output