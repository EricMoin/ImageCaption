import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
from transformers import GPT2Tokenizer

from dataset import ImageCaptioningDataset
from models import ImageCaptioningModel
from train import train_model

def main():
    # 创建必要的目录
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载GPT2 tokenizer
    print("Loading GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # 添加特殊token
    special_tokens = {
        'pad_token': '[PAD]',
        'bos_token': '[CLS]',
        'eos_token': '[SEP]',
        'unk_token': '[UNK]'
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 设置数据限制
    max_train_samples = 1024
    max_val_samples = 128
    batch_size = 16  # 减小批次大小
    num_epochs = 50
    max_len = 128    # GPT2上下文长度
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
    model = ImageCaptioningModel(vocab_size=len(tokenizer)).to(device)
    
    # 创建优化器和学习率调度器
    # 为不同组件设置不同的学习率
    optimizer = torch.optim.AdamW([
        {'params': model.vit.parameters(), 'lr': 1e-5},
        {'params': model.gpt.parameters(), 'lr': 2e-5},
        {'params': model.feature_mapping.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    
    # 使用余弦退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # 第一次重启的周期
        T_mult=2,  # 每次重启后周期翻倍
        eta_min=1e-6  # 最小学习率
    )
    
    # 如果有检查点，加载它
    checkpoint_path = 'checkpoints'
    checkpoint_file = f'{checkpoint_path}/best_model.pth'
    if os.path.exists(checkpoint_file):
        print(f"Found existing checkpoint at {checkpoint_file}")
        response = input("Model architecture has changed. Do you want to (1) start fresh or (2) try to load the checkpoint? [1/2]: ")
        if response == "2":
            try:
                print("Attempting to load checkpoint...")
                checkpoint = torch.load(checkpoint_file, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Successfully loaded compatible weights from checkpoint")
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
                print("Starting from scratch instead")
        else:
            print("Starting fresh with new model architecture")
            # 重命名旧的检查点文件
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            os.rename(checkpoint_file, f'{checkpoint_path}/old_model_{timestamp}.pth')
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