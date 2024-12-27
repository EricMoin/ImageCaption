import torch
import torch.nn as nn
import torchvision
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score.rouge_scorer import RougeScorer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import numpy as np

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

def evaluate_metrics(model, dataloader, device, vocab_idx2word):
    model.eval()
    references = []
    hypotheses = []
    raw_refs = {}  # 用于CIDEr和SPICE的原始参考
    raw_hyps = {}  # 用于CIDEr和SPICE的原始假设
    smoothing = SmoothingFunction().method1
    num_batches = len(dataloader)
    
    # 初始化评测器
    meteor_scorer = Meteor()
    rouge_scorer = RougeScorer(['rougeL'], use_stemmer=True)
    cider_scorer = Cider()
    spice_scorer = Spice()
    
    print(f"\nEvaluating on {len(dataloader.dataset)} samples in {num_batches} batches")
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # 生成描述
            generated_ids = generate_caption(model, images, device, vocab_idx2word)
            if generated_ids is None:
                print("Warning: Using empty generation for this batch")
                generated_ids = torch.full((images.size(0), 3), END_TOKEN, dtype=torch.long).to(device)
            
            # 转换为文本
            for idx, (gen_ids, cap_ids) in enumerate(zip(generated_ids, captions)):
                # 处理生成的描述
                pred_tokens = [vocab_idx2word[idx.item()] for idx in gen_ids 
                             if idx.item() not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]
                if not pred_tokens:
                    pred_tokens = ['<unk>']
                
                # 处理真实描述
                ref_tokens = [vocab_idx2word[idx.item()] for idx in cap_ids 
                            if idx.item() not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]
                if not ref_tokens:
                    ref_tokens = ['<unk>']
                
                # 为BLEU准备分词后的文本
                hypotheses.append(pred_tokens)
                references.append([ref_tokens])
                
                # 为其他指标准备原始文本
                current_idx = len(raw_refs)
                raw_refs[current_idx] = [' '.join(ref_tokens)]
                raw_hyps[current_idx] = [' '.join(pred_tokens)]
            
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
        return {
            'bleu1': 0.0, 'bleu4': 0.0, 'meteor': 0.0,
            'rouge_l': 0.0, 'cider': 0.0, 'spice': 0.0
        }
    
    try:
        # 计算BLEU分数
        bleu1 = corpus_bleu(references, hypotheses,
                           weights=(1.0, 0, 0, 0),
                           smoothing_function=smoothing)
        bleu4 = corpus_bleu(references, hypotheses,
                           weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=smoothing)
        
        # 计算METEOR分数
        meteor_score = meteor_scorer.compute_score(raw_refs, raw_hyps)[0]
        
        # 计算ROUGE-L分数
        rouge_scores = []
        for i in range(len(hypotheses)):
            scores = rouge_scorer.score(raw_refs[i][0], raw_hyps[i][0])
            rouge_scores.append(scores['rougeL'].fmeasure)
        rouge_l_score = np.mean(rouge_scores)
        
        # 计算CIDEr分数
        cider_score = cider_scorer.compute_score(raw_refs, raw_hyps)[0]
        
        # 计算SPICE分数
        spice_score = spice_scorer.compute_score(raw_refs, raw_hyps)[0]
        
    except Exception as e:
        print("Error calculating metrics:")
        print(f"Number of references: {len(references)}")
        print(f"Number of hypotheses: {len(hypotheses)}")
        print("Sample reference:", ' '.join(references[0][0]) if references else "No references")
        print("Sample hypothesis:", ' '.join(hypotheses[0]) if hypotheses else "No hypotheses")
        print(f"Error: {str(e)}")
        return {
            'bleu1': 0.0, 'bleu4': 0.0, 'meteor': 0.0,
            'rouge_l': 0.0, 'cider': 0.0, 'spice': 0.0
        }
    
    # 打印样本结果
    print("\nSample predictions:")
    for i in range(min(3, len(hypotheses))):
        print(f"\nReference: {' '.join(references[i][0])}")
        print(f"Generated: {' '.join(hypotheses[i])}")
    
    # 返回所有指标
    metrics = {
        'bleu1': bleu1,
        'bleu4': bleu4,
        'meteor': meteor_score,
        'rouge_l': rouge_l_score,
        'cider': cider_score,
        'spice': spice_score
    }
    
    # 打印所有指标
    print("\nEvaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric.upper()}: {score:.4f}")
    
    return metrics

def generate_caption(model, image, device, vocab_idx2word, max_len=200):
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
        base_temperature = 1.0  # 降低基础温度以生成更连贯的句子
        min_temperature = 0.5   # 保持较低的最小温度
        
        # 句子结构控制
        min_words_per_sentence = 8    # 增加每个句子的最小词数
        max_words_per_sentence = 20   # 设置每个句子的最大词数
        max_sentences = 5            # 增加最大句子数量
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
            temperature = max(min_temperature, base_temperature * (1 - progress * 0.3))
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
                elif words_since_period[b] >= max_words_per_sentence:
                    period_boost = min(5.0, (words_since_period[b] - max_words_per_sentence) * 0.5)
                    logits[b, :, PERIOD_TOKEN] += period_boost
                
                # 如果已经生成足够的句子，强制结束
                if sentence_count[b] >= max_sentences:
                    logits[b, :, :] = float('-inf')  # 禁用所有token
                    logits[b, :, END_TOKEN] = 0.0  # 只允许END token
                
                # 如果序列接近最大长度但还没有足够的句子，增加句号概率
                if i >= (max_len * 0.8) and sentence_count[b] < (max_sentences - 1):
                    logits[b, :, PERIOD_TOKEN] += 3.0
            
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
            if i >= max_len - 20:
                # 如果已经生成了至少一个��整的句子，可以结束
                if (sentence_count >= 1).all():
                    # 添加句号（如果当前不是句号）
                    if (generated[:, -1] != PERIOD_TOKEN).any():
                        period_token = torch.full((batch_size, 1), PERIOD_TOKEN, dtype=torch.long).to(device)
                        generated = torch.cat([generated, period_token], dim=1)
                    # 添加END token
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
    best_metrics = {
        'bleu1': 0.0,
        'bleu4': 0.0,
        'meteor': 0.0,
        'rouge_l': 0.0,
        'cider': 0.0,
        'spice': 0.0
    }
    best_loss = float('inf')
    patience = 5  # 增加耐心值
    no_improve_metrics = {metric: 0 for metric in best_metrics.keys()}  # 各指标没有改进的轮数
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
        
        # 计算所有评测指标
        metrics = evaluate_metrics(model, val_loader, device, vocab_idx2word)
        
        # 更新学习率（使用CIDEr作为主要指标）
        scheduler.step(metrics['cider'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # 检查是否有显著改善
        loss_improved = train_loss < (best_loss - min_delta)
        metrics_improved = {
            metric: score > (best_metrics[metric] + min_delta)
            for metric, score in metrics.items()
        }
        
        if loss_improved:
            best_loss = train_loss
            no_improve_loss = 0
        else:
            no_improve_loss += 1
        
        # 更新每个指标的改善状态
        for metric in best_metrics.keys():
            if metrics_improved[metric]:
                best_metrics[metric] = metrics[metric]
                no_improve_metrics[metric] = 0
            else:
                no_improve_metrics[metric] += 1
        
        # 如果任何指标有改善，保存模型
        if any(metrics_improved.values()):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'metrics': metrics,
                'vocab_size': vocab_size,
                'vocab_idx2word': vocab_idx2word,
                'initial_lr': initial_lr
            }, f'{checkpoint_path}/best_model.pth')
            print('\nNew best model saved!')
        
        # 打印改善状态
        print('\nMetrics improvement status:')
        print(f'Loss: {no_improve_loss} epochs without improvement')
        for metric, count in no_improve_metrics.items():
            print(f'{metric.upper()}: {count} epochs without improvement')
        
        print('\nBest scores:')
        print(f'Loss: {best_loss:.4f}')
        for metric, score in best_metrics.items():
            print(f'{metric.upper()}: {score:.4f}')
        
        # 早停条件：所有指标都没有改善
        if (no_improve_loss >= patience and 
            all(count >= patience for count in no_improve_metrics.values())):
            print('\nEarly stopping: No improvement in any metric')
            break
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
                'metrics': metrics,
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
             if idx.item() not in [0, 1, 2, 3]]  # 除特殊token
    
    return ' '.join(tokens) 