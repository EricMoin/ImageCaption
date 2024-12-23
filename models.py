import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, BertConfig

class ImageCaptioningModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 加载BERT模型和配置
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 使用ViT作为视觉编码器
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()  # 移除分类头
        
        # 冻结部分参数
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 只训练最后几层
        for name, param in self.vit.named_parameters():
            if 'encoder.layer.11' in name:
                param.requires_grad = True
        for name, param in self.bert.named_parameters():
            if any(layer in name for layer in ['layer.10', 'layer.11', 'pooler']):
                param.requires_grad = True
        
        # 视觉特征投影层
        self.visual_projection = nn.Sequential(
            nn.Linear(768, self.bert_config.hidden_size),
            nn.LayerNorm(self.bert_config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.bert_config.hidden_size, self.bert_config.vocab_size)
        )
        
    def forward(self, images, input_ids=None, attention_mask=None, labels=None):
        """
        Args:
            images: [batch_size, 3, 224, 224]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        """
        # 1. 视觉特征提取
        visual_features = self.vit(images)  # [batch_size, 768]
        visual_features = self.visual_projection(visual_features)  # [batch_size, hidden_size]
        
        # 2. 扩展视觉特征以匹配序列长度
        if input_ids is not None:
            seq_length = input_ids.size(1)
            visual_features = visual_features.unsqueeze(1).expand(-1, seq_length, -1)
            
            # 3. BERT处理
            # 将视觉特征作为BERT的嵌入
            bert_outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=visual_features,
                encoder_attention_mask=torch.ones_like(attention_mask),
                output_hidden_states=True,
                return_dict=True
            )
            
            # 4. 生成输出
            sequence_output = bert_outputs.last_hidden_state
            logits = self.output_layer(sequence_output)
            
            # 5. 计算损失
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.bert_config.vocab_size), labels.view(-1))
                
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': bert_outputs.hidden_states
            }
            
        else:
            return visual_features
    
    def generate(self, images, tokenizer, max_length=50, num_beams=4, temperature=1.0):
        """生成图像描述"""
        batch_size = images.size(0)
        device = images.device
        
        # 1. 获取视觉特征
        visual_features = self(images)  # [batch_size, hidden_size]
        
        # 2. 准备解码起始token
        input_ids = torch.full(
            (batch_size, 1),
            tokenizer.cls_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 3. 生成文本
        for _ in range(max_length - 1):
            # 创建attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # 扩展视觉特征
            seq_length = input_ids.size(1)
            curr_visual_features = visual_features.unsqueeze(1).expand(-1, seq_length, -1)
            
            # BERT处理
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=curr_visual_features,
                encoder_attention_mask=torch.ones_like(attention_mask),
                output_hidden_states=True,
                return_dict=True
            )
            
            # 生成下一个token
            sequence_output = outputs.last_hidden_state
            logits = self.output_layer(sequence_output)
            next_token_logits = logits[:, -1, :] / temperature
            
            # 采样下一个token
            next_token = torch.argmax(next_token_logits, dim=-1)
            
            # 添加到序列中
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # 检查是否生成了[SEP]
            if (next_token == tokenizer.sep_token_id).all():
                break
        
        return input_ids