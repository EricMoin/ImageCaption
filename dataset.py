import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
import torchvision.transforms as transforms

class ImageCaptioningDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None, tokenizer=None, max_length=50, max_samples=None):
        """
        Args:
            image_folder (str): 图像文件夹路径
            annotations_file (str): 包含图像描述的JSON文件路径
            transform: 图像转换函数
            tokenizer: BERT tokenizer
            max_length (int): 描述的最大长度
            max_samples (int): 最大样本数量
        """
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载注释
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 处理不同格式的JSON文件
        if isinstance(data, dict):
            # 如果是字典格式，转换为列表格式
            self.annotations = []
            for image_name, caption_data in data.items():
                if isinstance(caption_data, dict):
                    caption = caption_data.get('caption', '')
                else:
                    caption = caption_data
                self.annotations.append({
                    'image': image_name,
                    'caption': caption
                })
        elif isinstance(data, list):
            # 如果已经是列表格式，直接使用
            self.annotations = data
        else:
            # 如果是其他格式，尝试从annotations字段获取
            self.annotations = data.get('annotations', [])
            
        # 如果指定了最大样本数，则限制数据集大小
        if max_samples is not None:
            self.annotations = self.annotations[:max_samples]
            
        print(f"Loaded {len(self.annotations)} image-caption pairs")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # 获取图像路径和描述
        ann = self.annotations[idx]
        image_path = os.path.join(self.image_folder, ann['image'])
        caption = ann['caption']
        
        # 加载和预处理图像
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # 返回一个空的黑色图像
            image = torch.zeros(3, 224, 224)
        
        # 使用BERT tokenizer处理文本
        if self.tokenizer:
            encoding = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            caption_ids = encoding['input_ids'].squeeze(0)
        else:
            caption_ids = torch.tensor([])
        
        return image, caption_ids