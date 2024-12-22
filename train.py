import torch
import torch.nn as nn
import torchvision
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

# 定义特殊token的索引
START_TOKEN = 1
END_TOKEN = 2
PAD_TOKEN = 0
UNK_TOKEN = 3

def get_period_token_id(vocab_idx2word):
    """获取句号的token ID"""
    # 打印词表以便调试
    print("\nSearching for period token in vocabulary...")
    for idx, word in vocab_idx2word.items():
        if word in ['.', '。', '．']:  # 添加更多可能的句号形式
            print(f"Found period token: '{word}' with ID: {idx}")
            return idx
    
    # 如果找不到句号，使用默认的句号ID
    print("Warning: No period token found in vocabulary, using default token.")
    return END_TOKEN  # 暂时使用END token作为句号

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
            
            # 使用KL散度作为损失
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
            generated_ids = generate_caption(model, images, device, vocab_idx2word)
            if generated_ids is None:
                print("Warning: Using empty generation for this batch")
                # 使用空生成而不是跳过
                generated_ids = torch.full((images.size(0), 3), END_TOKEN, dtype=torch.long).to(device)
            
            # 转换为文本
            for gen_ids, cap_ids in zip(generated_ids, captions):
                # 处理生成的描述
                pred_tokens = [vocab_idx2word[idx.item()] for idx in gen_ids 
                             if idx.item() not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]
                if not pred_tokens:  # 如果生成的描述为空，添加一个占位符
                    pred_tokens = ['<unk>']
                hypotheses.append(pred_tokens)
                
                # 处理真实描述
                ref_tokens = [vocab_idx2word[idx.item()] for idx in cap_ids 
                            if idx.item() not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]
                if not ref_tokens:  # 如果参考描述为空，添加一个占位符
                    ref_tokens = ['<unk>']
                references.append([ref_tokens])
            
            # 打印评估进度和样本
            print(f'Evaluating batch [{batch_idx+1}/{num_batches}]')
            if batch_idx == 0:
                print("\nSample generations:")
                for i in range(min(3, len(hypotheses))):
                    print(f"Generated {i+1}: {' '.join(hypotheses[i])}")
                    print(f"Reference {i+1}: {' '.join(references[i][0])}")
    
    # 确保至少有一个有效的预测和参考
    if not hypotheses or not references:
        print("Warning: No valid predictions or references")
        return 0.0, 0.0
    
    try:
        # 计算BLEU分数
        bleu1 = corpus_bleu(references, hypotheses, 
                           weights=(1.0, 0, 0, 0),
                           smoothing_function=smoothing)
        bleu4 = corpus_bleu(references, hypotheses, 
                           weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=smoothing)
    except Exception as e:
        print("Error calculating BLEU score:")
        print(f"Number of references: {len(references)}")
        print(f"Number of hypotheses: {len(hypotheses)}")
        print("Sample reference:", references[0] if references else "No references")
        print("Sample hypothesis:", hypotheses[0] if hypotheses else "No hypotheses")
        print(f"Error: {str(e)}")
        return 0.0, 0.0
    
    # 打印样本结果
    print("\nSample predictions:")
    for i in range(min(3, len(hypotheses))):
        print(f"\nReference: {' '.join(references[i][0])}")
        print(f"Generated: {' '.join(hypotheses[i])}")
    
    return bleu1, bleu4

def generate_caption(model, image, device, vocab_idx2word, max_len=50):
    model.eval()
    
    # 获取句号的token ID
    PERIOD_TOKEN = None
    for idx, word in vocab_idx2word.items():
        if word == '.':
            PERIOD_TOKEN = idx
            break
    
    if PERIOD_TOKEN is None:
        print("Warning: Period token not found in vocabulary")
        return None
    
    with torch.no_grad():
        # 编码图像
        memory = model.encoder(image)
        
        # 准备起始token
        batch_size = image.size(0)
        start_token = torch.full((batch_size, 1), START_TOKEN, dtype=torch.long).to(device)
        
        generated = start_token
        
        # 动态调整温度参数
        base_temperature = 1.2  # 增加基础温度以提高多样性
        min_temperature = 0.6   # 增加最小温度以保持多样性
        
        # 句子结构控制
        min_words_per_sentence = 5  # 每个句子的最小词数
        max_sentences = 3       # 最大句子数量
        sentence_count = torch.zeros(batch_size, dtype=torch.long).to(device)
        words_since_period = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        for i in range(max_len - 1):  # 预留空间给END token
            # 生成mask
            tgt_mask = model.decoder.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            # 预测下一个token
            output = model.decoder(generated, memory, tgt_mask)
            logits = output[:, -1:]
            
            # 动态调整温度参数
            progress = i / max_len
            temperature = max(min_temperature, base_temperature * (1 - progress * 0.5))
            logits = logits / temperature
            
            # 更新句子统计
            last_token = generated[:, -1]
            sentence_count += (last_token == PERIOD_TOKEN).long()
            words_since_period += 1
            words_since_period *= (last_token != PERIOD_TOKEN).long()  # 如果是句号则重置
            
            # 调整token概率
            for b in range(batch_size):
                # 禁用句号直到达到最小词数
                if words_since_period[b] < min_words_per_sentence:
                    logits[b, :, PERIOD_TOKEN] = float('-inf')
                
                # 如果句子太长，增加句号概率
                elif words_since_period[b] >= min_words_per_sentence * 2:
                    period_boost = (words_since_period[b] - min_words_per_sentence) * 0.5
                    logits[b, :, PERIOD_TOKEN] += period_boost
                
                # 如果已经生成足够的句子，强制结束
                if sentence_count[b] >= max_sentences:
                    logits[b, :, :] = float('-inf')  # 禁用所有token
                    logits[b, :, END_TOKEN] = 0.0  # 只允许END token
            
            # 使用top-k和top-p采样
            top_k = 5
            top_p = 0.9
            
            # 首先进行top-k过滤
            top_k_logits, top_k_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            top_k_probs = torch.softmax(top_k_logits, dim=-1)
            
            # 然后进行top-p (nucleus) 采样
            sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 应用top-p过滤
            filtered_probs = sorted_probs.clone()
            filtered_probs[sorted_indices_to_remove] = 0
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            
            # 采样
            selected_indices = torch.multinomial(filtered_probs.squeeze(1), num_samples=1)
            next_token_idx = sorted_indices.squeeze(1).gather(1, selected_indices)
            next_token = top_k_indices.squeeze(1).gather(1, next_token_idx)
            
            # 添加预测的token
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否所有序列都生成了END token
            if (next_token == END_TOKEN).all():
                break
            
            # 如果序列太长，检查是否可以结束
            if i >= max_len - 10:
                # 如果已经生成了至少一个完整的句子，可以结束
                if (sentence_count >= 1).all():
                    end_token = torch.full((batch_size, 1), END_TOKEN, dtype=torch.long).to(device)
                    generated = torch.cat([generated, end_token], dim=1)
                    break
        
        # 确保所有序列都有合适的结束
        final_sequences = []
        for i in range(batch_size):
            seq = generated[i]
            # 如果序列没有以句号或END token结束，添加它们
            if seq[-1] != END_TOKEN:
                if seq[-1] != PERIOD_TOKEN:
                    seq = torch.cat([seq, torch.tensor([PERIOD_TOKEN], dtype=torch.long).to(device)])
                seq = torch.cat([seq, torch.tensor([END_TOKEN], dtype=torch.long).to(device)])
            final_sequences.append(seq)
        
        # 将所有序列填充到相同长度
        max_seq_len = max(len(seq) for seq in final_sequences)
        padded_sequences = []
        for seq in final_sequences:
            if len(seq) < max_seq_len:
                padding = torch.full((max_seq_len - len(seq),), PAD_TOKEN, 
                                  dtype=torch.long).to(device)
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        generated = torch.stack(padded_sequences)
    
    return generated

def train_model(model, train_loader, val_loader, vocab_idx2word, 
                num_epochs, criterion, optimizer, scheduler, device, checkpoint_path):
    best_bleu4 = 0
    best_loss = float('inf')
    patience = 5  # 增加耐心值
    no_improve_bleu = 0  # BLEU-4没有改善的轮数
    no_improve_loss = 0  # Loss没有改善的轮数
    min_delta = 1e-4  # 最小改善阈值
    
    # 保存词表信息
    vocab_size = len(vocab_idx2word)
    vocab_idx2word = {int(idx): word for idx, word in vocab_idx2word.items()}
    
    # 学习率预热
    warmup_epochs = 2
    warmup_factor = 0.1
    initial_lr = optimizer.param_groups[0]['lr']
    
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
                'initial_lr': initial_lr
            }, f'{checkpoint_path}/best_model.pth')
            print(f'\nNew best model saved! BLEU-4: {bleu4:.4f}')
        else:
            no_improve_bleu += 1
        
        # 打印改善状态
        print(f'\nNo improvement count - Loss: {no_improve_loss}, BLEU-4: {no_improve_bleu}')
        print(f'Best scores - Loss: {best_loss:.4f}, BLEU-4: {best_bleu4:.4f}')
        
        # 早停条件：只在性能没有改善时停止
        if no_improve_loss >= patience and no_improve_bleu >= patience:
            print(f'\nEarly stopping:')
            print(f'- Loss not improved for {no_improve_loss} epochs')
            print(f'- BLEU-4 not improved for {no_improve_bleu} epochs')
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
                'initial_lr': initial_lr
            }, f'{checkpoint_path}/checkpoint_epoch{epoch+1}.pth')
            
        print('-' * 50)

def generate_description(model, image_path, transform, vocab_idx2word, device):
    model.eval()
    
    # 加载和预处理图像
    image = torchvision.io.read_image(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    # 生成描述
    with torch.no_grad():
        generated_ids = generate_caption(model, image, device, vocab_idx2word)
        
    # 转换为文本
    tokens = [vocab_idx2word[idx.item()] for idx in generated_ids[0]
             if idx.item() not in [0, 1, 2, 3]]  # 移除特殊token
    
    return ' '.join(tokens) 