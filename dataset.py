import torch
from torch.utils.data import Dataset
import torchvision
import json
import spacy
import numpy as np
from itertools import islice

# 加载spacy英语模型
nlp = spacy.load('en_core_web_sm')

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None, max_length=50, max_samples=None):
        self.image_folder = image_folder
        self.max_length = max_length
        
        # 读取并可选地限制数据量
        with open(annotations_file, 'r') as f:
            full_annotations = json.load(f)
            if max_samples is not None:
                # 限制加载的数据量
                self.annotations = dict(islice(full_annotations.items(), max_samples))
                print(f"Loaded {len(self.annotations)} samples out of {len(full_annotations)}")
            else:
                self.annotations = full_annotations
                
        self.transform = transform
        
        # 构建词表
        self.vocab = {}
        self.build_vocabulary()
        
    def build_vocabulary(self):
        # 特殊标记
        self.vocab['<PAD>'] = 0
        self.vocab['<START>'] = 1
        self.vocab['<END>'] = 2
        self.vocab['<UNK>'] = 3
        
        # 使用spacy处理所有描述文本
        idx = 4
        for _, caption in self.annotations.items():
            if isinstance(nlp, type(spacy.load('en_core_web_sm'))):
                doc = nlp(caption.lower())
                tokens = [token.text for token in doc]
            else:
                tokens = nlp(caption)
                
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1
        
        print(f"Vocabulary size: {len(self.vocab)}")

    def pad_sequence(self, seq):
        # 如果序列长度超过max_length，截断它
        if len(seq) > self.max_length:
            return seq[:self.max_length]
        # 如果序列长度小于max_length，用PAD填充
        else:
            return seq + [self.vocab['<PAD>']] * (self.max_length - len(seq))

    def __getitem__(self, idx):
        # 获取图像名称和描述
        image_name = list(self.annotations.keys())[idx]
        caption = self.annotations[image_name]
        
        # 加载和转换图像
        image_path = f"{self.image_folder}/{image_name}"
        image = torchvision.io.read_image(image_path)
        if self.transform:
            image = self.transform(image)
            
        # 处理描述文本
        if isinstance(nlp, type(spacy.load('en_core_web_sm'))):
            doc = nlp(caption.lower())
            tokens = ['<START>'] + [token.text for token in doc] + ['<END>']
        else:
            tokens = ['<START>'] + nlp(caption.lower()) + ['<END>']
            
        caption_ids = []
        for token in tokens:
            if token in self.vocab:
                caption_ids.append(self.vocab[token])
            else:
                caption_ids.append(self.vocab['<UNK>'])
        
        # 填充或截断序列到固定长���
        caption_ids = self.pad_sequence(caption_ids)
                
        return image, torch.tensor(caption_ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.annotations)
    
    def get_vocab_size(self):
        return len(self.vocab) 