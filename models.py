import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class GraphNetwork(nn.Module):
    def __init__(self, d_model, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # 边缘特征维度
        edge_dim = d_model // 2
        
        # 节点更新网络
        self.node_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + edge_dim, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # 边缘更新网络
        self.edge_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, edge_dim),
                nn.LayerNorm(edge_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # 全局特征更新网络
        self.global_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model + edge_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # 空间位置编码
        self.spatial_pe = nn.Parameter(torch.randn(7, 7, d_model))
        
        # 最终的特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def build_graph(self, nodes, batch_size):
        """构建完全连接的图，包括空间位置信息"""
        device = nodes.device
        num_nodes = nodes.size(1)  # 49个节点(7x7)
        
        # 添加空间位置编码
        spatial_pe = self.spatial_pe.view(-1, nodes.size(-1))  # [49, d_model]
        nodes = nodes + spatial_pe.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 为每对节点创建边
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 计算节点对之间的边特征
                    edge_feat = self.edge_update[0](
                        torch.cat([nodes[:, i], nodes[:, j]], dim=-1)
                    )
                    edges.append(edge_feat)
        
        # 将边特征堆叠 [batch_size, num_edges, edge_dim]
        edges = torch.stack(edges, dim=1)
        
        return nodes, edges

    def forward(self, x):
        batch_size = x.size(0)
        nodes = x
        
        # 构建初始图
        nodes, edges = self.build_graph(nodes, batch_size)
        
        # 多层图网络处理
        for layer in range(self.num_layers):
            # 1. 更新边特征
            edge_features = []
            edge_idx = 0
            num_nodes = nodes.size(1)
            
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        # 获取当前边特征
                        edge_feat = edges[:, edge_idx]
                        
                        # 更新边特征
                        new_edge_feat = self.edge_update[layer](
                            torch.cat([nodes[:, i], nodes[:, j]], dim=-1)
                        )
                        edge_features.append(new_edge_feat)
                        edge_idx += 1
            
            edges = torch.stack(edge_features, dim=1)
            
            # 2. 更新节点特征
            new_nodes = []
            for i in range(num_nodes):
                # 收集与当前节点相连的所有边特征
                connected_edges = []
                edge_idx = 0
                for j in range(num_nodes):
                    if i != j:
                        connected_edges.append(edges[:, edge_idx])
                        edge_idx += 1
                
                # 聚合边特征
                edge_aggr = torch.mean(torch.stack(connected_edges, dim=1), dim=1)
                
                # 更新节点特征
                node_input = torch.cat([nodes[:, i], edge_aggr], dim=-1)
                new_node = self.node_update[layer](node_input)
                new_nodes.append(new_node)
            
            nodes = torch.stack(new_nodes, dim=1)
            
            # 3. 更新全局特征
            global_edge_feat = torch.mean(edges, dim=1)
            global_node_feat = torch.mean(nodes, dim=1)
            global_feat = self.global_update[layer](
                torch.cat([global_node_feat, global_edge_feat], dim=-1)
            )
            
            # 将全局特征广播到所有节点
            nodes = nodes + global_feat.unsqueeze(1)
        
        # 融合原始特征和图网络处理后的特征
        output = self.feature_fusion(
            torch.cat([x, nodes], dim=-1)
        )
        
        return output

class GridEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        # 1. 特征提取backbone
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # 选择性地训练backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[-2:].parameters():
            param.requires_grad = True
        
        # 2. 特征投影层
        self.feature_projection = nn.Conv2d(2048, d_model, kernel_size=1)
        
        # 3. 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 4. 图网络处理区域特征
        self.graph_network = GraphNetwork(d_model, num_layers)
        
        # 5. 最终的特征归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 1. 提取网格特征
        features = self.backbone(x)
        
        # 2. 特征投影
        features = self.feature_projection(features)
        
        # 3. 重塑为序列
        batch_size = features.size(0)
        features = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)
        
        # 4. 添加位置编码
        features = self.pos_encoding(features)
        
        # 5. 图网络处理
        features = self.graph_network(features)
        
        # 6. 特征归一化
        features = self.norm(features)
        
        return features

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        
        # 网格编码器
        self.grid_encoder = GridEncoder(d_model, nhead, num_encoder_layers)
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
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
            num_layers=num_decoder_layers
        )
        
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
        # 编码图像特征
        memory = self.grid_encoder(src)
        
        # 词嵌入和位置编码
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        
        # Transformer解码
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 生成词概率
        output = self.output_layer(output)
        
        return output