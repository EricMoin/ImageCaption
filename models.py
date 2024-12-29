import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel, GPT2Config

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768):
        super().__init__()
        
        # 视觉编码器 (ViT)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # 冻结ViT的大部分参数
        for param in self.vit.parameters():
            param.requires_grad = False
        # 只训练最后几层
        for param in self.vit.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # GPT2配置和模型 - 启用交叉注意力
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        gpt2_config.add_cross_attention = True  # 启用交叉注意力
        gpt2_config.use_cache = True
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
        self.gpt.resize_token_embeddings(vocab_size)  # 调整词表大小
        
        # 冻结GPT2的部分参数
        for param in self.gpt.parameters():
            param.requires_grad = False
        # 只训练交叉注意力层、最后几层和词嵌入
        trainable_layers = ['wte', 'crossattention', 'h.9', 'h.10', 'h.11', 'ln_f']
        for name, param in self.gpt.named_parameters():
            if any(layer in name for layer in trainable_layers):
                param.requires_grad = True
        
        # 特征映射层
        self.feature_mapping = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
    def forward(self, images, input_ids=None, attention_mask=None, labels=None):
        batch_size = images.size(0)
        device = images.device
        
        # 1. 提取视觉特征
        vision_outputs = self.vit(images, output_hidden_states=True)
        image_features = vision_outputs.last_hidden_state
        
        # 2. 映射特征维度
        image_features = self.feature_mapping(image_features)
        
        if input_ids is not None:
            # 3. GPT2生成
            outputs = self.gpt(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=image_features,
                encoder_attention_mask=torch.ones(batch_size, image_features.size(1), device=device),
                labels=labels,
                use_cache=True,
                return_dict=True
            )
            
            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'hidden_states': outputs.hidden_states
            }
        
        return image_features
    
    def generate(self, images, tokenizer, max_length=50, temperature=1.0, min_length=10, top_k=10, top_p=0.9):
        """生成图像描述"""
        batch_size = images.size(0)
        device = images.device
        
        # 1. 获取视觉特征
        vision_outputs = self.vit(images, output_hidden_states=True)
        image_features = vision_outputs.last_hidden_state
        image_features = self.feature_mapping(image_features)
        
        # 2. 准备起始token
        input_ids = torch.full(
            (batch_size, 1),
            tokenizer.bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 3. 使用GPT2的生成功能
        output_sequences = self.gpt.generate(
            input_ids=input_ids,
            encoder_hidden_states=image_features,
            encoder_attention_mask=torch.ones(batch_size, image_features.size(1), device=device),
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            no_repeat_ngram_size=3
        )
        
        return output_sequences