import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np

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
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 设置数据限制 - 增加训练样本数量
    max_train_samples = 1000  # 增加到1000个训练样本
    max_val_samples = 200     # 增加到200个验证样本
    
    print("Loading datasets...")
    # 创建数据集
    train_dataset = ImageCaptioningDataset(
        image_folder='data/images',
        annotations_file='data/train_captions.json',
        transform=transform,
        max_length=50,
        max_samples=max_train_samples
    )
    
    val_dataset = ImageCaptioningDataset(
        image_folder='data/images',
        annotations_file='data/test_captions.json',
        transform=transform,
        max_length=50,
        max_samples=max_val_samples
    )
    
    print("\nCreating data loaders...")
    # 创建数据加载器 - 减小batch size以增加更新次数
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16,  # 减小batch size
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=16,  # 减小batch size
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建词表的反向映射（索引到单词）
    vocab_idx2word = {idx: word for word, idx in train_dataset.vocab.items()}
    
    print("\nInitializing model...")
    # 初始化模型 - 添加dropout
    model = ImageCaptioningModel(
        vocab_size=len(train_dataset.vocab),
        d_model=512
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<PAD>'])
    
    # 使用带有权重衰减的Adam优化器来增加正则化
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01,  # 添加L2正则化
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
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
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
    
    print("\nStarting training...")
    # 训练模型 - 增加训练轮数
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_idx2word=vocab_idx2word,
        num_epochs=50,  # 增加训练轮数
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,  # 添加学习率调度器
        device=device,
        checkpoint_path=checkpoint_path
    )

if __name__ == '__main__':
    # 设置随机种子以确保可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行主程序
    main()