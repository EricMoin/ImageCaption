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
    def __init__(self, grid_size=14, num_layers=6, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 使用ViT作为特征提取器
        self.vit = torchvision.models.vit_l_16(weights='DEFAULT')
        
        # 修改ViT的head为Identity，这样输出原始特征
        self.vit.heads = nn.Identity()
        
        # 冻结大部分预训练参数
        for name, param in self.vit.named_parameters():
            if 'encoder.layer.23' not in name:  # 只训练最后一个block
                param.requires_grad = False
        
        # 特征转换层
        self.feature_projection = nn.Sequential(
            nn.Linear(1024, d_model * 2),  # ViT-L输出维度为1024
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, grid_size * grid_size, d_model))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
    def forward(self, x):
        # 1. 通过ViT提取特征
        x = self.vit(x)  # 返回特征，shape: [batch_size, 1024]
        
        # 2. 重塑特征为序列形式
        batch_size = x.size(0)
        x = x.view(batch_size, 1, -1).expand(-1, 196, -1)  # shape: [batch_size, 196, 1024]
        
        # 3. 特征投影
        x = self.feature_projection(x)  # shape: [batch_size, 196, d_model]
        
        # 4. 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # 5. 通过额外的Transformer编码器
        x = self.transformer_encoder(x)
        
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        
        # 添加输入映射层
        self.input_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, vocab_size)
        )
        
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, tgt, memory, tgt_mask=None):
        # 词嵌入和dropout
        x = self.embedding(tgt)
        x = self.embedding_dropout(x)
        
        # 位置编码
        x = self.pos_embedding(x.transpose(0, 1)).transpose(0, 1)
        
        # 输入映射
        x = self.input_projection(x)
        
        # 通过Transformer解码器
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        
        # 输出映射
        output = self.output_projection(output)
        
        return output

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512):
        super().__init__()
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=8,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=8,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2
        )
        
    def forward(self, img, tgt, tgt_mask=None):
        memory = self.encoder(img)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output