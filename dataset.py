import torch
from torch.utils.data import Dataset
import torchvision
import json
import nltk
import numpy as np
from itertools import islice
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None, max_length=50, max_samples=None, use_cache=True):
        self.image_folder = image_folder
        self.max_length = max_length
        self.transform = transform
        self.use_cache = use_cache
        
        # 读取并可选地限制数据量
        print("Loading annotations...")
        with open(annotations_file, 'r') as f:
            data = json.load(f)
            # 确保数据格式正确
            if isinstance(data, dict):
                full_annotations = {k: v['caption'] if isinstance(v, dict) else v 
                                 for k, v in data.items()}
            else:
                full_annotations = {item['image']: item['caption'] 
                                 for item in data['annotations']}
            
            if max_samples is not None:
                # 限制加载的数据量
                self.annotations = dict(islice(full_annotations.items(), max_samples))
                print(f"Loaded {len(self.annotations)} samples out of {len(full_annotations)}")
            else:
                self.annotations = full_annotations
        
        # 构建或加载词表
        print("Building/loading vocabulary...")
        self.vocab = {}
        self._init_special_tokens()
        
        # 尝试从缓存加载词表
        vocab_cache_file = self._get_vocab_cache_path(annotations_file, max_samples)
        if os.path.exists(vocab_cache_file):
            print(f"Loading vocabulary from cache: {vocab_cache_file}")
            self._load_vocab_from_cache(vocab_cache_file)
        else:
            print("Building vocabulary from scratch...")
            self.build_vocabulary()
            # 保存词表到缓存
            self._save_vocab_to_cache(vocab_cache_file)
        
        # 初始化图像缓存
        self.cache = {}
        if self.use_cache:
            print("Pre-loading images...")
            self._preload_images()
    
    def _get_vocab_cache_path(self, annotations_file, max_samples):
        """生成词表缓存文件路径"""
        # 使用注释文件名和样本数量生成唯一的缓存文件名
        base_name = os.path.splitext(os.path.basename(annotations_file))[0]
        cache_dir = os.path.join(os.path.dirname(annotations_file), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 添加版本号到缓存文件名
        vocab_version = "v1"  # 更改此版本号会使所有旧缓存失效
        return os.path.join(cache_dir, f'{base_name}_vocab_{max_samples if max_samples else "full"}_{vocab_version}.pkl')
    
    def _save_vocab_to_cache(self, cache_file):
        """保存词表到缓存文件"""
        print(f"Saving vocabulary to cache: {cache_file}")
        cache_data = {
            'vocab': self.vocab,
            'version': "v1",  # 保存版本信息
            'timestamp': os.path.getmtime(cache_file) if os.path.exists(cache_file) else None
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_vocab_from_cache(self, cache_file):
        """从缓存文件加载词表"""
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # 检查版本
        if isinstance(cache_data, dict) and cache_data.get('version') == "v1":
            self.vocab = cache_data['vocab']
        else:
            # 如果版本不匹配，删除旧缓存并重新构建
            os.remove(cache_file)
            print("Cache version mismatch, rebuilding vocabulary...")
            self.build_vocabulary()
            self._save_vocab_to_cache(cache_file)
            return
        
        print(f"Loaded vocabulary of size: {len(self.vocab)}")
    
    def _init_special_tokens(self):
        """初始化特殊标记"""
        special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>', '.']  # 添加句号作为特殊标记
        for idx, token in enumerate(special_tokens):
            self.vocab[token] = idx
    
    def _clean_text(self, text):
        """清理和标准化文本"""
        # 转换为小写
        text = text.lower()
        # 保留句号，但移除其他特殊字符
        text = re.sub(r'[^\w\s\.]', ' ', text)  # 修改正则表达式以保留句号
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def build_vocabulary(self):
        """使用NLTK构建词表"""
        # 使用集合来存储唯一的词
        word_set = set()
        
        # 获取停用词（可选）
        # stop_words = set(stopwords.words('english'))
        
        # 批处理大小
        batch_size = 100
        all_captions = [str(caption) for caption in self.annotations.values()]
        total_batches = (len(all_captions) + batch_size - 1) // batch_size
        
        # 批量处理文本
        for i in range(0, len(all_captions), batch_size):
            if i % batch_size == 0:
                print(f"Processing batch {i//batch_size + 1}/{total_batches}")
            
            batch = all_captions[i:i + batch_size]
            for text in batch:
                # 清理文本，保留句号
                clean_text = self._clean_text(text)
                # 分词
                tokens = word_tokenize(clean_text)
                # 添加到词集合
                word_set.update(tokens)
        
        # 将收集到的词添加到词表中
        current_idx = len(self.vocab)
        for word in sorted(word_set):  # 排序以确保词表顺序一致
            if word not in self.vocab:
                self.vocab[word] = current_idx
                current_idx += 1
        
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # 验证词表完整性
        assert self.vocab['<PAD>'] == 0
        assert self.vocab['<START>'] == 1
        assert self.vocab['<END>'] == 2
        assert self.vocab['<UNK>'] == 3
        assert self.vocab['.'] == 4  # 确保句号在词表中
        
        # 打印词表中的标点符号
        print("\nPunctuation marks in vocabulary:")
        for word, idx in self.vocab.items():
            if word in ['.', ',', '!', '?', ';', ':']:
                print(f"'{word}': {idx}")
    
    def _preload_images(self):
        """预加载所有图像到内存"""
        total_images = len(self.annotations)
        for idx, image_name in enumerate(self.annotations.keys()):
            if idx % 100 == 0:
                print(f"Pre-loading images: {idx}/{total_images}")
            
            image_path = f"{self.image_folder}/{image_name}"
            try:
                image = torchvision.io.read_image(image_path)
                if self.transform:
                    image = self.transform(image)
                self.cache[image_name] = image
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
                continue
        print("Finished pre-loading images")

    def __getitem__(self, idx):
        # 获取图像名称和描述
        image_name = list(self.annotations.keys())[idx]
        caption = self.annotations[image_name]
        
        # 确保caption是字符串
        if not isinstance(caption, str):
            caption = str(caption)
        
        # 加载图像（从缓存或磁盘）
        if self.use_cache and image_name in self.cache:
            image = self.cache[image_name]
        else:
            image_path = f"{self.image_folder}/{image_name}"
            try:
                image = torchvision.io.read_image(image_path)
                if self.transform:
                    image = self.transform(image)
                if self.use_cache:
                    self.cache[image_name] = image
            except Exception as e:
                print(f"Error loading image {image_name}: {e}")
                # 返回一个空图像
                image = torch.zeros((3, 224, 224))
        
        # 处理描述文本
        try:
            # 清理和分词
            clean_text = self._clean_text(caption)
            tokens = ['<START>'] + word_tokenize(clean_text) + ['<END>']
            
            # 转换为ID
            caption_ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
            
            # 填充或截断序列到固定长度
            if len(caption_ids) > self.max_length:
                caption_ids = caption_ids[:self.max_length-1] + [self.vocab['<END>']]
            else:
                caption_ids.extend([self.vocab['<PAD>']] * (self.max_length - len(caption_ids)))
                    
            return image, torch.tensor(caption_ids, dtype=torch.long)
        except Exception as e:
            print(f"Error processing caption: {caption}")
            print(f"Error: {e}")
            # 返回一个空的caption
            empty_caption = [self.vocab['<START>'], self.vocab['<END>']]
            empty_caption.extend([self.vocab['<PAD>']] * (self.max_length - 2))
            return image, torch.tensor(empty_caption, dtype=torch.long)
    
    def __len__(self):
        return len(self.annotations)
    
    def get_vocab_size(self):
        return len(self.vocab)
        
    def decode_caption(self, caption_ids):
        """将ID序列解码为文本"""
        # 使用列表推导式提高效率
        words = [word for id in caption_ids
                if (word := {v: k for k, v in self.vocab.items()}.get(id.item())) not in 
                ['<START>', '<PAD>', '<END>', '<UNK>']]
        return ' '.join(words)