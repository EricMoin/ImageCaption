import torch
import torchvision
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

def train_epoch(model, dataloader, criterion, optimizer, device, max_len=50):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    print(f"Training on {len(dataloader.dataset)} samples in {num_batches} batches")
    
    # 梯度裁剪阈值
    grad_clip = 1.0
    
    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        
        # 准备目标序列（移除最后一个token）和目标输出（移除第一个token）
        tgt_input = captions[:, :-1]
        tgt_output = captions[:, 1:]
        
        # 创建mask
        tgt_mask = model.decoder.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
        # 前向传播
        optimizer.zero_grad()
        try:
            output = model(images, tgt_input, tgt_mask)
            
            # 计算损失，添加标签平滑
            smooth_factor = 0.1
            n_classes = output.size(-1)
            one_hot = torch.zeros_like(output).scatter(
                2, tgt_output.unsqueeze(-1), 1
            )
            one_hot = one_hot * (1 - smooth_factor) + (smooth_factor / n_classes)
            
            # 使用KL散度作为损失函数
            log_prb = torch.log_softmax(output, dim=-1)
            loss = -(one_hot * log_prb).sum(dim=-1).mean()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # 更新参数
            optimizer.step()
            
            total_loss += loss.item()
            
            # 对于小数据集，每个batch都打印进度
            print(f'Batch [{batch_idx+1}/{num_batches}], '
                  f'Samples [{(batch_idx+1)*len(images)}/{len(dataloader.dataset)}], '
                  f'Loss: {loss.item():.4f}')
                
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}:")
            print(f"Image shape: {images.shape}")
            print(f"Target input shape: {tgt_input.shape}")
            print(f"Mask shape: {tgt_mask.shape}")
            raise e
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_bleu(model, dataloader, device, vocab_idx2word):
    model.eval()
    references = []
    hypotheses = []
    smoothing = SmoothingFunction().method1
    num_batches = len(dataloader)
    
    print(f"\nEvaluating on {len(dataloader.dataset)} samples in {num_batches} batches")
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # 生成描述
            generated_ids = generate_caption(model, images, device)
            
            # 转换为文本
            for gen_ids, cap_ids in zip(generated_ids, captions):
                # 处理生成的描述
                pred_tokens = [vocab_idx2word[idx.item()] for idx in gen_ids 
                             if idx.item() not in [0, 1, 2, 3]]  # 移除特殊token
                if not pred_tokens:  # 如果生成的描述为空，添加一个占位符
                    pred_tokens = ['<unk>']
                hypotheses.append(pred_tokens)
                
                # 处理真实描述
                ref_tokens = [vocab_idx2word[idx.item()] for idx in cap_ids 
                            if idx.item() not in [0, 1, 2, 3]]
                if not ref_tokens:  # 如果参考描述为空，添加一个占位符
                    ref_tokens = ['<unk>']
                references.append([ref_tokens])
            
            # 打印评估进度
            print(f'Evaluating batch [{batch_idx+1}/{num_batches}]')
    
    try:
        # 计算BLEU分数，使用平滑函数
        bleu1 = corpus_bleu(references, hypotheses, 
                           weights=(1.0, 0, 0, 0),
                           smoothing_function=smoothing)
        bleu4 = corpus_bleu(references, hypotheses, 
                           weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=smoothing)
        
        # 打印一些样本结果
        print("\nSample predictions:")
        for i in range(min(3, len(hypotheses))):
            print(f"\nReference: {' '.join(references[i][0])}")
            print(f"Generated: {' '.join(hypotheses[i])}")
            
    except Exception as e:
        print("Error calculating BLEU score:")
        print(f"Number of references: {len(references)}")
        print(f"Number of hypotheses: {len(hypotheses)}")
        print("Sample reference:", references[0] if references else "No references")
        print("Sample hypothesis:", hypotheses[0] if hypotheses else "No hypotheses")
        raise e
    
    return bleu1, bleu4

def generate_caption(model, image, device, max_len=50):
    model.eval()
    
    with torch.no_grad():
        # 编码像
        memory = model.encoder(image)
        
        # 准备起始token
        batch_size = image.size(0)
        start_token = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)  # <START> token
        
        generated = start_token
        
        for _ in range(max_len - 1):
            # 生成mask
            tgt_mask = model.decoder.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            # 预测下一个token
            output = model.decoder(generated, memory, tgt_mask)
            
            # 添加温度参数和top-k采样
            temperature = 0.7
            logits = output[:, -1:] / temperature
            top_k = 5
            
            # 获取top-k的概率和索引
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            # 根据概率采样
            selected_indices = torch.multinomial(top_probs.squeeze(1), num_samples=1)
            next_token = top_indices.squeeze(1).gather(1, selected_indices)
            
            # 添加预测的token
            generated = torch.cat([generated, next_token], dim=1)
            
            # 如果生成了结束token，就停止
            if (next_token == 2).all():  # <END> token
                break
    
    return generated

def train_model(model, train_loader, val_loader, vocab_idx2word, 
                num_epochs, criterion, optimizer, scheduler, device, checkpoint_path):
    best_bleu4 = 0
    best_loss = float('inf')
    patience = 3  # 降低耐心值
    no_improve_bleu = 0  # BLEU-4没有改善的轮数
    no_improve_loss = 0  # Loss没有改善的轮数
    min_delta = 1e-4  # 最小改善阈值
    
    # 保存词表信息
    vocab_size = len(vocab_idx2word)
    vocab_idx2word = {int(idx): word for idx, word in vocab_idx2word.items()}
    
    # 学习率预热
    warmup_epochs = 2  # 减少预热轮数
    warmup_factor = 0.1
    initial_lr = optimizer.param_groups[0]['lr']  # 保存初始学习率
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 学习率预热
        if epoch < warmup_epochs:
            factor = warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr * factor
                print(f'Warmup learning rate: {param_group["lr"]:.6f}')
        
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'\nAverage Training Loss: {train_loss:.4f}')
        
        # 计算BLEU分数
        bleu1, bleu4 = evaluate_bleu(model, val_loader, device, vocab_idx2word)
        print(f'\nBLEU Scores:')
        print(f'BLEU-1: {bleu1:.4f}')
        print(f'BLEU-4: {bleu4:.4f}')
        
        # 更新学习率
        scheduler.step(bleu4)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # 检查是否有显著改善
        loss_improved = train_loss < (best_loss - min_delta)
        bleu_improved = bleu4 > (best_bleu4 + min_delta)
        
        if loss_improved:
            best_loss = train_loss
            no_improve_loss = 0
        else:
            no_improve_loss += 1
            
        if bleu_improved:
            best_bleu4 = bleu4
            no_improve_bleu = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'bleu4': bleu4,
                'vocab_size': vocab_size,
                'vocab_idx2word': vocab_idx2word,
                'initial_lr': initial_lr  # 保存初始学习率
            }, f'{checkpoint_path}/best_model.pth')
            print(f'\nNew best model saved! BLEU-4: {bleu4:.4f}')
        else:
            no_improve_bleu += 1
        
        # 打印改善状态
        print(f'\nNo improvement count - Loss: {no_improve_loss}, BLEU-4: {no_improve_bleu}')
        print(f'Best scores - Loss: {best_loss:.4f}, BLEU-4: {best_bleu4:.4f}')
        
        # 早停条件：
        # 1. Loss连续3轮没有改善
        # 2. BLEU-4连续3轮没有改善
        # 3. 学习率已经很小
        if (no_improve_loss >= patience and no_improve_bleu >= patience) or \
           current_lr < 1e-6:
            print(f'\nEarly stopping:')
            print(f'- Loss not improved for {no_improve_loss} epochs')
            print(f'- BLEU-4 not improved for {no_improve_bleu} epochs')
            print(f'- Current learning rate: {current_lr}')
            break
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'bleu4': bleu4,
                'vocab_size': vocab_size,
                'vocab_idx2word': vocab_idx2word,
                'initial_lr': initial_lr  # 保存初始学习率
            }, f'{checkpoint_path}/checkpoint_epoch{epoch+1}.pth')
            
        print('-' * 50)

def generate_description(model, image_path, transform, vocab_idx2word, device):
    model.eval()
    
    # 加载和预处理图像
    image = torchvision.io.read_image(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    # 生成描述
    with torch.no_grad():
        generated_ids = generate_caption(model, image, device)
        
    # 转换为文本
    tokens = [vocab_idx2word[idx.item()] for idx in generated_ids[0]
             if idx.item() not in [0, 1, 2, 3]]  # 移除特殊token
    
    return ' '.join(tokens) 