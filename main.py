import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from transformers import BertTokenizer

from dataset import ImageCaptioningDataset
from models import ImageCaptioningModel
from train import train_model, generate_description

def main():
    # 创建必要的目录
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载BERT tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 设置数据限制
    max_train_samples = None  # 不限制训练样本数量
    max_val_samples = None    # 不限制验证样本数量
    batch_size = 32
    num_epochs = 50
    max_len = 200
    num_workers = 0
    
    print("Loading datasets...")
    # 创建数据集
    print("Loading training dataset...")
    train_dataset = ImageCaptioningDataset(
        image_folder='data/images',
        annotations_file='data/train_captions.json',
        transform=transform,
        tokenizer=tokenizer,
        max_length=max_len,
        max_samples=max_train_samples
    )
    
    print("Loading validation dataset...")
    val_dataset = ImageCaptioningDataset(
        image_folder='data/images',
        annotations_file='data/test_captions.json',
        transform=transform,
        tokenizer=tokenizer,
        max_length=max_len,
        max_samples=max_val_samples
    )
    
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print("\nInitializing model...")
    # 初始化模型
    model = ImageCaptioningModel().to(device)
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        [
            {'params': model.vit.parameters(), 'lr': 1e-4},
            {'params': model.bert.parameters(), 'lr': 5e-5},
            {'params': [p for n, p in model.named_parameters() 
                       if not any(m in n for m in ['vit', 'bert'])], 
             'lr': 1e-4}
        ],
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # 如果有检查点，加载它
    checkpoint_path = 'checkpoints'
    checkpoint_file = f'{checkpoint_path}/best_model.pth'
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Successfully loaded checkpoint")
    else:
        print("No checkpoint found, starting from scratch")
    
    print("\nStarting training...")
    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer=tokenizer
    )

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    main()