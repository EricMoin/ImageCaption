import torch
import torchvision
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from collections import defaultdict
import math
from nltk.corpus import wordnet
import spacy
import en_core_web_sm

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
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        
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

def compute_cider(refs, hypos):
    """计算CIDEr-D分数"""
    def compute_vec(tokens):
        vec = defaultdict(float)
        length = len(tokens)
        for i in range(length - 3):
            gram = ' '.join(tokens[i:i + 4])  # 使用4-gram
            vec[gram] += 1
        for gram, count in vec.items():
            vec[gram] = count / float(length - 3)
        return vec
    
    # 计算IDF
    doc_freq = defaultdict(float)
    num_refs = len(refs)
    for ref_tokens in refs:
        vec = compute_vec(ref_tokens[0])  # 使用第一个参考描述
        for gram in vec:
            doc_freq[gram] += 1
    
    # 计算IDF权重
    for gram, freq in doc_freq.items():
        doc_freq[gram] = math.log(num_refs / freq)
    
    # 计算每个句子对的CIDEr分数
    scores = []
    for hypo_tokens, ref_list in zip(hypos, refs):
        # 计算候选句子的向量
        hypo_vec = compute_vec(hypo_tokens)
        
        # 计算参考句子的向量
        ref_vecs = [compute_vec(ref) for ref in ref_list]
        
        # 应用IDF权重
        for gram in hypo_vec:
            hypo_vec[gram] *= doc_freq[gram]
        for ref_vec in ref_vecs:
            for gram in ref_vec:
                ref_vec[gram] *= doc_freq[gram]
        
        # 计算余弦相似度
        score = 0.0
        for ref_vec in ref_vecs:
            # 计算分子
            numerator = sum(hypo_vec[gram] * ref_vec[gram] for gram in set(hypo_vec) & set(ref_vec))
            
            # 计算分母
            hypo_norm = math.sqrt(sum(val ** 2 for val in hypo_vec.values()))
            ref_norm = math.sqrt(sum(val ** 2 for val in ref_vec.values()))
            
            if hypo_norm > 0 and ref_norm > 0:
                score += numerator / (hypo_norm * ref_norm)
        
        score /= len(ref_vecs)  # 平均分数
        scores.append(score * 10.0)  # 乘以10以匹配原始CIDEr-D的比例
    
    return np.mean(scores)

def compute_spice(refs, hypos):
    """计算SPICE分数"""
    nlp = en_core_web_sm.load()
    
    def extract_scene_graph(text):
        """提取场景图（简化版本）"""
        doc = nlp(text)
        graph = {
            'entities': set(),
            'relations': set(),
            'attributes': set()
        }
        
        # 提取实体和属性
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                graph['entities'].add(token.text)
            elif token.pos_ in ['ADJ', 'ADV']:
                graph['attributes'].add(token.text)
        
        # 提取关系
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                if token.head.pos_ == 'VERB':
                    subj = token.text
                    verb = token.head.text
                    for child in token.head.children:
                        if child.dep_ in ['dobj', 'pobj']:
                            obj = child.text
                            graph['relations'].add((subj, verb, obj))
        
        return graph
    
    scores = []
    for hypo, ref_list in zip(hypos, refs):
        hypo_text = ' '.join(hypo)
        hypo_graph = extract_scene_graph(hypo_text)
        
        ref_scores = []
        for ref in ref_list:
            ref_text = ' '.join(ref)
            ref_graph = extract_scene_graph(ref_text)
            
            # 计算F1分数
            prec_e = len(hypo_graph['entities'] & ref_graph['entities']) / len(hypo_graph['entities']) if hypo_graph['entities'] else 0
            recall_e = len(hypo_graph['entities'] & ref_graph['entities']) / len(ref_graph['entities']) if ref_graph['entities'] else 0
            f1_e = 2 * prec_e * recall_e / (prec_e + recall_e) if (prec_e + recall_e) > 0 else 0
            
            prec_r = len(hypo_graph['relations'] & ref_graph['relations']) / len(hypo_graph['relations']) if hypo_graph['relations'] else 0
            recall_r = len(hypo_graph['relations'] & ref_graph['relations']) / len(ref_graph['relations']) if ref_graph['relations'] else 0
            f1_r = 2 * prec_r * recall_r / (prec_r + recall_r) if (prec_r + recall_r) > 0 else 0
            
            prec_a = len(hypo_graph['attributes'] & ref_graph['attributes']) / len(hypo_graph['attributes']) if hypo_graph['attributes'] else 0
            recall_a = len(hypo_graph['attributes'] & ref_graph['attributes']) / len(ref_graph['attributes']) if ref_graph['attributes'] else 0
            f1_a = 2 * prec_a * recall_a / (prec_a + recall_a) if (prec_a + recall_a) > 0 else 0
            
            # 综合分数
            ref_scores.append((f1_e + f1_r + f1_a) / 3)
        
        scores.append(max(ref_scores))  # 使用最高分数
    
    return np.mean(scores)

def evaluate_metrics(model, dataloader, device, vocab_idx2word):
    model.eval()
    references = []
    hypotheses = []
    smoothing = SmoothingFunction().method1
    num_batches = len(dataloader)
    
    # 初始化ROUGE评分器
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print(f"\nEvaluating on {len(dataloader.dataset)} samples in {num_batches} batches")
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # 生成描述
            generated_ids = generate_caption(model, images, device, vocab_idx2word)
            
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
        # 计算BLEU分数
        bleu1 = corpus_bleu(references, hypotheses, 
                           weights=(1.0, 0, 0, 0),
                           smoothing_function=smoothing)
        bleu4 = corpus_bleu(references, hypotheses, 
                           weights=(0.25, 0.25, 0.25, 0.25),
                           smoothing_function=smoothing)
        
        # 计算METEOR分数
        meteor_scores = []
        for hyp, ref in zip(hypotheses, references):
            score = meteor_score(ref, hyp)
            meteor_scores.append(score)
        meteor = np.mean(meteor_scores)
        
        # 计算ROUGE-L分数
        rouge_scores = []
        for hyp, ref in zip(hypotheses, references):
            score = rouge_scorer_obj.score(' '.join(hyp), ' '.join(ref[0]))
            rouge_scores.append(score['rougeL'].fmeasure)
        rouge_l = np.mean(rouge_scores)
        
        # 计算CIDEr-D分数
        cider = compute_cider(references, hypotheses)
        
        # 计算SPICE分数
        spice = compute_spice(references, hypotheses)
        
        # 打印一些样本结果
        print("\nSample predictions:")
        for i in range(min(3, len(hypotheses))):
            print(f"\nReference: {' '.join(references[i][0])}")
            print(f"Generated: {' '.join(hypotheses[i])}")
            
    except Exception as e:
        print("Error calculating metrics:")
        print(f"Number of references: {len(references)}")
        print(f"Number of hypotheses: {len(hypotheses)}")
        print("Sample reference:", references[0] if references else "No references")
        print("Sample hypothesis:", hypotheses[0] if hypotheses else "No hypotheses")
        raise e
    
    return {
        'bleu1': bleu1,
        'bleu4': bleu4,
        'meteor': meteor,
        'rouge_l': rouge_l,
        'cider': cider,
        'spice': spice
    }

def generate_caption(model, image, device, vocab_idx2word, max_len=200,max_sentences=10,min_words_per_sentence=5):
    model.eval()
    
    with torch.no_grad():
        # 通过ViT提取特征
        features = model.vit(image)  # [batch_size, 768]
        
        # 投影到d_model维度
        features = model.feature_projection(features)  # [batch_size, d_model]
        
        # 扩展特征维度以匹配Transformer的输入要求
        memory = features.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # 准备起始token
        batch_size = image.size(0)
        start_token = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)  # <START> token
        
        generated = start_token
        
        # 跟踪句子状态
        words_since_period = torch.zeros(batch_size, dtype=torch.long).to(device)  # 自上一个句号后��单词数
        sentences_generated = torch.zeros(batch_size, dtype=torch.long).to(device)  # 已生成的句子数
        
        for i in range(max_len - 1):
            # 生成mask
            tgt_mask = model.generate_square_subsequent_mask(generated.size(1)).to(device)
            
            # 词嵌入
            tgt = model.embedding(generated)  # [batch_size, seq_len, d_model]
            
            # 位置编码
            tgt = tgt.transpose(0, 1)  # [seq_len, batch_size, d_model]
            tgt = model.pos_encoder(tgt)
            tgt = tgt.transpose(0, 1)  # [batch_size, seq_len, d_model]
            
            # Transformer解码
            output = model.transformer_decoder(
                tgt,
                memory,
                tgt_mask=tgt_mask
            )
            
            # 生成词概率
            output = model.output_layer(output)  # [batch_size, seq_len, vocab_size]
            
            # 如果接近最大长度，强制选择END token
            if i >= max_len - 3:
                next_token = torch.full((batch_size, 1), 2, dtype=torch.long).to(device)  # <END> token
            else:
                # 添加温度参数和top-k采样
                temperature = 0.7
                logits = output[:, -1:] / temperature
                
                # 调整句号和END token的概率
                for b in range(batch_size):
                    # 如果当前句子太短，禁止使用句号
                    if words_since_period[b] < min_words_per_sentence:
                        logits[b, :, 4] = float('-inf')  # 4是句号的索引
                    
                    # 如果当前句子足够长，增加句号的概率
                    elif words_since_period[b] >= min_words_per_sentence:
                        logits[b, :, 4] += 1.0  # 增加句号的logit值
                    
                    # 如果已经生成了足够多的句子，增加END token的概率
                    if sentences_generated[b] >= max_sentences - 1 and words_since_period[b] >= min_words_per_sentence:
                        logits[b, :, 2] += 2.0  # 增加END token的logit值
                    
                    # 如果序列长度超过90%，增加句号和END token的概率
                    if i >= max_len * 0.9:
                        logits[b, :, 4] += 2.0  # 增加句号的概率
                        if words_since_period[b] >= min_words_per_sentence:
                            logits[b, :, 2] += 3.0  # 增加END token的概率
                
                # top-k采样
                top_k = 5
                top_probs, top_indices = torch.topk(torch.softmax(logits, dim=-1), k=top_k, dim=-1)
                
                # 根据概率采样
                selected_indices = torch.multinomial(top_probs.squeeze(1), num_samples=1)
                next_token = top_indices.squeeze(1).gather(1, selected_indices)
            
            # 添加预测的token
            generated = torch.cat([generated, next_token], dim=1)
            
            # 更新句子状态
            for b in range(batch_size):
                token_id = next_token[b].item()
                if token_id == 4:  # 句号
                    sentences_generated[b] += 1
                    words_since_period[b] = 0
                elif token_id not in [0, 1, 2, 3, 4]:  # 不是特殊token或句号
                    words_since_period[b] += 1
            
            # 如果生成了结束token或达到最大句子数，就停止
            if (next_token == 2).all() or (sentences_generated >= max_sentences).all():
                break
        
        # 确保所有序列都以句号和END token结束
        final_sequences = []
        for b in range(batch_size):
            seq = generated[b]
            if seq[-1] != 2:  # 如果最后一个token不是END
                if words_since_period[b] > 0:  # 如果当前句子还没有结束
                    # 添加句号
                    seq = torch.cat([seq, torch.tensor([4], dtype=torch.long).to(device)])
                # 添加END token
                seq = torch.cat([seq, torch.tensor([2], dtype=torch.long).to(device)])
            final_sequences.append(seq)
        
        # 找到最长公共子序列的长度
        max_length = max(seq.size(0) for seq in final_sequences)
        
        # 将所有序列填充到相同长度
        padded_sequences = []
        for seq in final_sequences:
            if seq.size(0) < max_length:
                padding = torch.full((max_length - seq.size(0),), 0, dtype=torch.long).to(device)
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        # 堆叠所有序列
        generated = torch.stack(padded_sequences)
    
    return generated

def train_model(model, train_loader, val_loader, vocab_idx2word, 
                num_epochs, criterion, optimizer, scheduler, device, checkpoint_path):
    best_metrics = {
        'bleu1': 0,
        'bleu4': 0,
        'meteor': 0,
        'rouge_l': 0,
        'cider': 0,
        'spice': 0
    }
    best_loss = float('inf')
    patience = 3  # 降低耐心值
    no_improve_metrics = {metric: 0 for metric in best_metrics.keys()}  # 各指标没有改善的轮数
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
        
        # 计算评估指标
        metrics = evaluate_metrics(model, val_loader, device, vocab_idx2word)
        print('\nValidation Metrics:')
        for metric, value in metrics.items():
            print(f'{metric.upper()}: {value:.4f}')
        
        # 更新学习率
        scheduler.step(metrics['bleu4'])
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # 检查是否有显著改善
        loss_improved = train_loss < (best_loss - min_delta)
        metrics_improved = {metric: value > (best_metrics[metric] + min_delta)
                          for metric, value in metrics.items()}
        
        if loss_improved:
            best_loss = train_loss
            no_improve_loss = 0
        else:
            no_improve_loss += 1
        
        # 更新最佳指标和计数器
        for metric, improved in metrics_improved.items():
            if improved:
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
        print('\nNo improvement count:')
        print(f'Loss: {no_improve_loss}')
        for metric, count in no_improve_metrics.items():
            print(f'{metric.upper()}: {count}')
        
        print('\nBest scores:')
        print(f'Loss: {best_loss:.4f}')
        for metric, value in best_metrics.items():
            print(f'{metric.upper()}: {value:.4f}')
        
        # 早停条件：
        # 1. Loss连续3轮没有改善
        # 2. 所有指标连续3轮没有改善
        # 3. 学习率已经很小
        if (no_improve_loss >= patience and 
            all(count >= patience for count in no_improve_metrics.values())) or \
           current_lr < 1e-6:
            print(f'\nEarly stopping:')
            print(f'- Loss not improved for {no_improve_loss} epochs')
            for metric, count in no_improve_metrics.items():
                print(f'- {metric.upper()} not improved for {count} epochs')
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
        generated_ids = generate_caption(model, image, device)
        
    # 转换为文本
    tokens = [vocab_idx2word[idx.item()] for idx in generated_ids[0]
             if idx.item() not in [0, 1, 2, 3]]  # 移除特殊token
    
    return ' '.join(tokens) 