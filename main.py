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
    max_train_samples = 1024  # 增加到1000个训练样本
    max_val_samples = 256     # 增加到200个验证样本
    batch_size = 64
    num_epochs = 50
    max_len = 80                                 
    num_workers = 0
    
    print("Loading datasets...")
    # 创建数据集
    print("Loading training dataset...")
    train_dataset = ImageCaptioningDataset(
        image_folder='data/images',
        annotations_file='data/train_captions.json',
        transform=transform,
        max_length=max_len,
        max_samples=max_train_samples,
        use_cache=True  # 启用缓存
    )
    
    print("Loading validation dataset...")
    val_dataset = ImageCaptioningDataset(
        image_folder='data/images',
        annotations_file='data/test_captions.json',
        transform=transform,
        max_length=max_len,
        max_samples=max_val_samples,
        use_cache=True  # 启用缓存
    )
    
    print("\nCreating data loaders...")
    # 创建数据加载器 - 优化worker数量和内存使用
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 减少worker数量
        pin_memory=False,  # 关闭pin_memory以减少内存使用
        persistent_workers=False  # 关闭持久化workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # 减少worker数量
        pin_memory=False,  # 关闭pin_memory以减少内存使用
        persistent_workers=False  # 关闭持久化workers
    )
    
    # 创建词表的反向映射（索引到单词）
    vocab_idx2word = {idx: word for word, idx in train_dataset.vocab.items()}
    
    print("\nInitializing model...")
    # 初始化模型 - 添加dropout
    model = ImageCaptioningModel(
        vocab_size=len(train_dataset.vocab),
        max_length=max_len,
        d_model=512
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<PAD>'])
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    # 使用余弦退火学习率调度器
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
        
        # 检查词表大小是否匹配
        checkpoint_vocab_size = checkpoint.get('vocab_size')
        current_vocab_size = len(train_dataset.vocab)
        
        if checkpoint_vocab_size != current_vocab_size:
            print(f"\nWarning: Vocabulary size mismatch!")
            print(f"Checkpoint vocabulary size: {checkpoint_vocab_size}")
            print(f"Current vocabulary size: {current_vocab_size}")
            print("Creating new model with current vocabulary size...")
            
            # 重新创建模型
            model = ImageCaptioningModel(
                vocab_size=current_vocab_size,
                d_model=512
            ).to(device)
            
            # 重新初始化优化器和调度器
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=2,
                verbose=True
            )
        else:
            # 词表大小匹配，可以加载检查点
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Successfully loaded checkpoint")
        
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Starting from epoch {start_epoch + 1}")
        print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    else:
        print("No checkpoint found, starting from scratch")
    
    print("\nStarting training...")
    # 训练模型 - 增加训练轮数
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_idx2word=vocab_idx2word,
        num_epochs=num_epochs,  # 增加训练轮数
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