import torch
import torch.nn as nn
import torchvision.models as models
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

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6):
        super().__init__()
        
        # 使用ViT作为视觉编码器
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # 移除分类头
        self.vit.heads = nn.Identity()
        
        # 冻结大部分ViT参数
        for name, param in self.vit.named_parameters():
            if 'encoder.layer.11' not in name:  # 只训练最后一个block
                param.requires_grad = False
        
        # 特征投影层
        self.feature_projection = nn.Sequential(
            nn.Linear(768, d_model * 2),  # ViT-B输出维度为768
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
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
        # 通过ViT提取特征
        features = self.vit(images)  # [batch_size, 768]
        
        # 投影到d_model维度
        features = self.feature_projection(features)  # [batch_size, d_model]
        
        # 扩展特征维度以匹配Transformer的输入要求
        memory = features.unsqueeze(1)  # [batch_size, 1, d_model]
        
        if captions is not None:
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