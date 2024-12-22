import torch
import torch.nn as nn
import torchvision
import math

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, q, k, v, attn_mask=None):
        # 多头注意力
        attn_output = self.attention(q, k, v, attn_mask=attn_mask)[0]
        x = self.norm(q + self.dropout(attn_output))
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_output))
        
        return x

class GridEncoder(nn.Module):
    def __init__(self, grid_size=7, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        # CNN特征提取器
        resnet = torchvision.models.resnet50(weights='DEFAULT')
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])
        
        # 冻结CNN参数
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        self.grid_size = grid_size
        
        # 特征转换层
        self.feature_projection = nn.Sequential(
            nn.Linear(2048, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, d_model))
        
        # 多层多头注意力
        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # 1. 通过CNN提取特征
        features = self.cnn(x)
        
        # 2. 调整特征图大小为固定的网格大小
        B, C, H, W = features.shape
        features = nn.functional.adaptive_avg_pool2d(features, (self.grid_size, self.grid_size))
        
        # 3. 重塑为序列形式
        features = features.reshape(B, C, -1).permute(0, 2, 1)
        
        # 4. 投影到所需维度
        features = self.feature_projection(features)
        
        # 5. 添加位置编码
        features = features + self.pos_embed
        
        # 6. 多层自注意力处理
        for layer in self.attention_layers:
            features = layer(features, features, features)
        
        return features

class GridDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=3, dropout=0.1, max_length=80):
        super().__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, d_model))  # 使用max_length参数
        
        # 多层自注意力
        self.self_attention_layers = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # 多层交叉注意力
        self.cross_attention_layers = nn.ModuleList([
            MultiHeadAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, vocab_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(self, tgt, memory, tgt_mask=None):
        # 1. 词嵌入
        x = self.embedding(tgt)
        
        # 2. 添加位置编码
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len]
        x = self.dropout(x)
        
        # 3. 生成mask（如果没有提供）
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 4. 多层自注意力
        for layer in self.self_attention_layers:
            x = layer(x, x, x, attn_mask=tgt_mask)
        
        # 5. 多层交叉注意力
        for layer in self.cross_attention_layers:
            x = layer(x, memory, memory)
        
        # 6. 输出预测
        output = self.output(x)
        
        return output

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, nhead=8,num_layers=3,d_model=512, max_length=80):
        super().__init__()
        self.encoder = GridEncoder(
            grid_size=7,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=0.1
        )
        self.decoder = GridDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=0.1,
            max_length=max_length  # 传递max_length参数
        )
        
    def forward(self, img, tgt, tgt_mask=None):
        # 1. 编码图像得到网格特征
        memory = self.encoder(img)
        
        # 2. 解码生成文本
        output = self.decoder(tgt, memory, tgt_mask)
        
        return output