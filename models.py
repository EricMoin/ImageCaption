import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, max_length=50):
        super().__init__()
        
        # CNN backbone for grid features
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc
        
        # 冻结backbone的大部分参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        # 只训练最后两个blocks
        for param in list(self.backbone.parameters())[-20:]:  # 最后两个blocks的参数
            param.requires_grad = True
            
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Conv2d(2048, d_model, kernel_size=1),  # 将ResNet的2048通道映射到d_model
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer解码器
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, vocab_size)
        )
        
        # 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
                
    def generate_square_subsequent_mask(self, sz):
        """生成注意力掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, images, captions=None, tgt_mask=None):
        """
        Args:
            images: [batch_size, 3, 224, 224]
            captions: [batch_size, seq_len]
            tgt_mask: [seq_len, seq_len]
        """
        # 提取网格特征
        features = self.backbone(images)  # [batch_size, 2048, 7, 7]
        
        # 投影到d_model维度
        features = self.feature_projection(features)  # [batch_size, d_model, 7, 7]
        
        # 重塑为序列
        batch_size = features.size(0)
        features = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)  # [batch_size, 49, d_model]
        
        # Transformer编码
        memory = self.transformer_encoder(features)  # [batch_size, 49, d_model]
        
        if self.training and captions is not None:
            # 训练模式
            # 词嵌入
            tgt = self.embedding(captions)  # [batch_size, seq_len, d_model]
            
            # 位置编码
            tgt = tgt.transpose(0, 1)  # [seq_len, batch_size, d_model]
            tgt = self.pos_encoder(tgt)
            tgt = tgt.transpose(0, 1)  # [batch_size, seq_len, d_model]
            
            # Transformer解码
            output = self.transformer_decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask
            )
            
            # 生成词概率
            output = self.output_layer(output)  # [batch_size, seq_len, vocab_size]
            
        else:
            # 推理模式
            output = memory
            
        return output