import torch
import torchvision
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from collections import defaultdict
import re
from nltk.corpus import wordnet
import spacy
from itertools import chain
import math
import os

def train_epoch(model, dataloader, criterion, optimizer, device):
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

def compute_meteor(references, hypotheses):
    """计算METEOR分数"""
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Warning: Error loading NLTK data: {e}")
        pass
    
    scores = []
    for hyp, refs in zip(hypotheses, references):
        # 确保输入是分词后的列表
        hyp_tokens = hyp if isinstance(hyp, list) else nltk.word_tokenize(hyp)
        refs_tokens = [ref if isinstance(ref, list) else nltk.word_tokenize(' '.join(ref)) 
                      for ref in refs]
        
        # 计算每个参考与假设之间的METEOR分数
        try:
            score = max(nltk.translate.meteor_score.single_meteor_score(ref, hyp_tokens)
                       for ref in refs_tokens)
        except Exception as e:
            print(f"Error computing METEOR score: {e}")
            print(f"Hypothesis: {hyp_tokens}")
            print(f"References: {refs_tokens}")
            score = 0.0
        
        scores.append(score)
    
    # 返回平均分数
    return np.mean(scores) if scores else 0.0

def compute_rouge_l(references, hypotheses):
    """计算ROUGE-L分数"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_str = ' '.join(hyp)
        max_score = 0
        for ref in refs:
            ref_str = ' '.join(ref)
            score = scorer.score(ref_str, hyp_str)['rougeL'].fmeasure
            max_score = max(max_score, score)
        scores.append(max_score)
    return np.mean(scores)

def compute_cider(references, hypotheses, n_gram=4):
    """计算CIDEr-D分数"""
    def build_ngrams(tokens, n):
        ngrams = defaultdict(int)
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] += 1
        return ngrams
    
    def compute_tf_idf(refs_ngrams, hyp_ngrams, doc_freq, ref_len):
        epsilon = 1e-12
        tf_idf_scores = []
        
        for n in range(1, n_gram + 1):
            # 计算TF
            ref_tf = defaultdict(float)
            hyp_tf = defaultdict(float)
            
            # 计算参考文本的TF
            for ngram, count in refs_ngrams[n-1].items():
                ref_tf[ngram] = count / max(ref_len, 1)
            
            # 计算假设文本的TF
            hyp_len = sum(hyp_ngrams[n-1].values())
            for ngram, count in hyp_ngrams[n-1].items():
                hyp_tf[ngram] = count / max(hyp_len, 1)
            
            # 计算IDF
            num_refs = len(doc_freq)
            idf = {}
            for ngram in set(chain(ref_tf.keys(), hyp_tf.keys())):
                df = doc_freq.get(ngram, 0)
                idf[ngram] = math.log(num_refs / (df + epsilon))
            
            # 计算余弦相似度
            numerator = 0
            ref_norm = 0
            hyp_norm = 0
            
            # 所有n-gram的并集
            all_ngrams = set(chain(ref_tf.keys(), hyp_tf.keys()))
            
            for ngram in all_ngrams:
                ref_tfidf = ref_tf[ngram] * idf.get(ngram, 0)
                hyp_tfidf = hyp_tf[ngram] * idf.get(ngram, 0)
                
                numerator += ref_tfidf * hyp_tfidf
                ref_norm += ref_tfidf * ref_tfidf
                hyp_norm += hyp_tfidf * hyp_tfidf
            
            denom = math.sqrt(max(ref_norm, epsilon)) * math.sqrt(max(hyp_norm, epsilon))
            score = numerator / denom if denom > epsilon else 0
            
            tf_idf_scores.append(score)
        
        return np.mean(tf_idf_scores)
    
    # 计算文档频率
    doc_freq = defaultdict(float)
    for refs in references:
        doc_ngrams = set()  # 使用集合来确保每个文档中的n-gram只计算一次
        for ref in refs:
            for n_size in range(1, n_gram + 1):
                ngrams = build_ngrams(ref, n_size)
                doc_ngrams.update(ngrams.keys())
        for ngram in doc_ngrams:
            doc_freq[ngram] += 1
    
    scores = []
    for hyp, refs in zip(hypotheses, references):
        # 为假设构建n-grams
        hyp_ngrams = []
        for i in range(n_gram):
            hyp_ngrams.append(build_ngrams(hyp, i + 1))
        
        # 为每个参考构建n-grams并计算分数
        ref_scores = []
        for ref in refs:
            ref_ngrams = []
            for i in range(n_gram):
                ref_ngrams.append(build_ngrams(ref, i + 1))
            score = compute_tf_idf(ref_ngrams, hyp_ngrams, doc_freq, len(ref))
            ref_scores.append(score)
        
        # 使用最高分数
        scores.append(max(ref_scores) if ref_scores else 0)
    
    # 返回平均分数，乘以10作为最终的CIDEr-D分数
    final_score = np.mean(scores) * 10.0 if scores else 0.0
    return final_score

def compute_spice(references, hypotheses):
    """计算SPICE分数"""
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        import subprocess
        subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
        nlp = spacy.load('en_core_web_sm')
    
    def extract_scene_graph(text):
        doc = nlp(text)
        entities = set()
        relations = set()
        attributes = set()
        
        # 提取实体和属性
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                entities.add(token.text.lower())
                # 添加形容词修饰
                for child in token.children:
                    if child.pos_ == 'ADJ':
                        attributes.add((token.text.lower(), child.text.lower()))
        
        # 提取关系
        for token in doc:
            if token.pos_ == 'VERB':
                subj = None
                obj = None
                for child in token.children:
                    if child.dep_ == 'nsubj':
                        subj = child.text.lower()
                    elif child.dep_ in ['dobj', 'pobj']:
                        obj = child.text.lower()
                if subj and obj:
                    relations.add((subj, token.text.lower(), obj))
        
        return entities, relations, attributes
    
    def compute_f1(ref_graph, hyp_graph):
        ref_entities, ref_relations, ref_attributes = ref_graph
        hyp_entities, hyp_relations, hyp_attributes = hyp_graph
        
        # 计算实体F1
        common_entities = len(ref_entities & hyp_entities)
        if len(ref_entities) == 0 and len(hyp_entities) == 0:
            entity_f1 = 1.0
        elif len(ref_entities) == 0 or len(hyp_entities) == 0:
            entity_f1 = 0.0
        else:
            precision = common_entities / len(hyp_entities)
            recall = common_entities / len(ref_entities)
            entity_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # 计算关系F1
        common_relations = len(ref_relations & hyp_relations)
        if len(ref_relations) == 0 and len(hyp_relations) == 0:
            relation_f1 = 1.0
        elif len(ref_relations) == 0 or len(hyp_relations) == 0:
            relation_f1 = 0.0
        else:
            precision = common_relations / len(hyp_relations)
            recall = common_relations / len(ref_relations)
            relation_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # 计算属性F1
        common_attributes = len(ref_attributes & hyp_attributes)
        if len(ref_attributes) == 0 and len(hyp_attributes) == 0:
            attribute_f1 = 1.0
        elif len(ref_attributes) == 0 or len(hyp_attributes) == 0:
            attribute_f1 = 0.0
        else:
            precision = common_attributes / len(hyp_attributes)
            recall = common_attributes / len(ref_attributes)
            attribute_f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        # 综合分数
        return (entity_f1 + relation_f1 + attribute_f1) / 3
    
    scores = []
    for hyp, refs in zip(hypotheses, references):
        hyp_text = ' '.join(hyp)
        hyp_graph = extract_scene_graph(hyp_text)
        
        ref_scores = []
        for ref in refs:
            ref_text = ' '.join(ref)
            ref_graph = extract_scene_graph(ref_text)
            score = compute_f1(ref_graph, hyp_graph)
            ref_scores.append(score)
        
        scores.append(max(ref_scores))
    
    return np.mean(scores)

def evaluate_metrics(model, dataloader, device, vocab_idx2word):
    """评估所有指标"""
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
            if (batch_idx + 1) % 5 == 0:
                print(f'Evaluating batch [{batch_idx+1}/{num_batches}]')
    
    metrics = {}
    try:
        # 计算BLEU分数
        metrics['bleu1'] = corpus_bleu(references, hypotheses, 
                                     weights=(1.0, 0, 0, 0),
                                     smoothing_function=smoothing)
        metrics['bleu4'] = corpus_bleu(references, hypotheses, 
                                     weights=(0.25, 0.25, 0.25, 0.25),
                                     smoothing_function=smoothing)
    except Exception as e:
        print(f"Error calculating BLEU scores: {e}")
        metrics['bleu1'] = 0.0
        metrics['bleu4'] = 0.0
    
    try:
        # 计算METEOR分数
        metrics['meteor'] = compute_meteor(references, hypotheses)
    except Exception as e:
        print(f"Error calculating METEOR score: {e}")
        metrics['meteor'] = 0.0
    
    try:
        # 计算ROUGE-L分数
        metrics['rouge_l'] = compute_rouge_l(references, hypotheses)
    except Exception as e:
        print(f"Error calculating ROUGE-L score: {e}")
        metrics['rouge_l'] = 0.0
    
    try:
        # 计算CIDEr分数
        metrics['cider'] = compute_cider(references, hypotheses)
    except Exception as e:
        print(f"Error calculating CIDEr score: {e}")
        metrics['cider'] = 0.0
    
    try:
        # 计算SPICE分数
        metrics['spice'] = compute_spice(references, hypotheses)
    except Exception as e:
        print(f"Error calculating SPICE score: {e}")
        metrics['spice'] = 0.0
    
    # 打印一些样本结果
    print("\nSample predictions:")
    for i in range(min(3, len(hypotheses))):
        print(f"\nReference: {' '.join(references[i][0])}")
        print(f"Generated: {' '.join(hypotheses[i])}")
    
    # 打印所有指标
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def generate_caption(model, image, device, vocab_idx2word, max_len=200, max_sentences=5, min_words_per_sentence=5):
    model.eval()
    
    with torch.no_grad():
        # 提取网格特征
        features = model.backbone(image)  # [batch_size, 2048, 7, 7]
        
        # 投影到d_model维度
        features = model.feature_projection(features)  # [batch_size, d_model, 7, 7]
        
        # 重塑为序列
        batch_size = features.size(0)
        features = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)  # [batch_size, 49, d_model]
        
        # Transformer编码
        memory = model.transformer_encoder(features)  # [batch_size, 49, d_model]
        
        # 准备起始token
        start_token = torch.full((batch_size, 1), 1, dtype=torch.long).to(device)  # <START> token
        
        generated = start_token
        
        # 跟踪句子状态
        words_since_period = torch.zeros(batch_size, dtype=torch.long).to(device)
        sentences_generated = torch.zeros(batch_size, dtype=torch.long).to(device)
        
        for i in range(max_len - 3):  # 预留空间给句号和END token
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
            
            # 获最后一个时间步的输出
            logits = output[:, -1:] / 0.7  # 使用温度参数
            
            # 动态调整token概率
            for b in range(batch_size):
                # 如果当前句子太短，禁止使用句号
                if words_since_period[b] < min_words_per_sentence:
                    logits[b, :, 4] = float('-inf')  # 4是句号的索引
                
                # 如果当前句子足够长，增加句号的概率
                elif words_since_period[b] >= min_words_per_sentence:
                    logits[b, :, 4] += 1.0
                
                # 如果已经生成了足够多的句子，增加END token的概率
                if sentences_generated[b] >= max_sentences - 1 and words_since_period[b] >= min_words_per_sentence:
                    logits[b, :, 2] += 2.0  # 2是END token的索引
                
                # 如果序列长度超过90%，增加句号和END token的概率
                if i >= (max_len - 5):
                    logits[b, :, 4] += 2.0  # 增加句号的概���
                    if words_since_period[b] >= min_words_per_sentence:
                        logits[b, :, 2] += 3.0  # 增加END token的概率
            
            # top-k采样
            top_k = 5
            top_probs, top_indices = torch.topk(torch.softmax(logits, dim=-1), k=top_k, dim=-1)
            
            # 采样下一个token
            selected_indices = torch.multinomial(top_probs.squeeze(1), num_samples=1)
            next_token = top_indices.squeeze(1).gather(1, selected_indices)
            
            # 添加预测的token
            generated = torch.cat([generated, next_token], dim=1)
            
            # 更新句子状态
            for b in range(batch_size):
                token = next_token[b].item()
                if token == 4:  # 句号
                    sentences_generated[b] += 1
                    words_since_period[b] = 0
                elif token not in [0, 1, 2]:  # 不是特殊token
                    words_since_period[b] += 1
            
            # 检查是否应该停止生成
            if (next_token == 2).all() or (sentences_generated >= max_sentences).all():
                break
        
        # 确保所有序列都以句号和END token束
        final_sequences = []
        for b in range(batch_size):
            seq = generated[b]
            # 如果序列太长，截断它
            if len(seq) > max_len - 2:  # 为句号和END token预留空间
                seq = seq[:max_len - 2]
            
            # 如果最后一个token不是END且当前句子未结束
            if seq[-1] != 2 and words_since_period[b] > 0:
                # 添加句号
                seq = torch.cat([seq, torch.tensor([4], device=device)])
            
            # 如果最后一个token不是END
            if seq[-1] != 2:
                # 添加END token
                seq = torch.cat([seq, torch.tensor([2], device=device)])
            
            # 确保序列长度不超过max_len
            if len(seq) > max_len:
                seq = seq[:max_len]
            
            final_sequences.append(seq)
        
        # 找到最长序列的长度
        max_seq_len = max(len(seq) for seq in final_sequences)
        
        # 所有序列填充到相同长度
        padded_sequences = []
        for seq in final_sequences:
            if len(seq) < max_seq_len:
                padding = torch.full((max_seq_len - len(seq),), 0, dtype=torch.long, device=device)
                seq = torch.cat([seq, padding])
            padded_sequences.append(seq)
        
        # 堆叠所有序列
        generated = torch.stack(padded_sequences)
    
    return generated

def train_model(model, train_loader, val_loader, vocab_idx2word, 
                num_epochs, criterion, optimizer, scheduler, device, checkpoint_path):
    # 初始化所有指标的最佳值
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
    no_improve_metrics = {metric: 0 for metric in best_metrics.keys()}  # 各指标没有改��的轮数
    no_improve_loss = 0  # Loss没有改善的轮数
    min_delta = 1e-4  # 最小改善阈值
    
    # 保存词表信息
    vocab_size = len(vocab_idx2word)
    vocab_idx2word = {int(idx): word for idx, word in vocab_idx2word.items()}
    
    # 学习率预热
    warmup_epochs = 2  # 减少预热轮数
    warmup_factor = 0.1
    initial_lr = optimizer.param_groups[0]['lr']  # 保存初始学习率
    
    # 检查是否有检查点
    checkpoint_file = f'{checkpoint_path}/best_model.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        
        # 检查词表大小是否匹配
        checkpoint_vocab_size = checkpoint.get('vocab_size')
        if checkpoint_vocab_size == vocab_size:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('loss', float('inf'))
            # 恢复最佳指标
            if 'best_metrics' in checkpoint:
                best_metrics.update(checkpoint['best_metrics'])
            print("Successfully loaded checkpoint")
            print("Best metrics from checkpoint:")
            for metric, value in best_metrics.items():
                print(f"{metric}: {value:.4f}")
        else:
            print(f"Warning: Vocabulary size mismatch! Expected {vocab_size}, got {checkpoint_vocab_size}")
            print("Starting from scratch with new vocabulary")
    
    for epoch in range(start_epoch, num_epochs):
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
        
        # 计算所有评估指标
        metrics = evaluate_metrics(model, val_loader, device, vocab_idx2word)
        print('\nValidation Metrics:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')
        
        # 更新学习率 - 使用综合指标
        composite_score = (metrics['bleu4'] + metrics['meteor'] + metrics['rouge_l'] + 
                         metrics['cider']/10 + metrics['spice']) / 5
        scheduler.step(composite_score)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # 检查是否有显著改善
        loss_improved = train_loss < (best_loss - min_delta)
        metrics_improved = {
            metric: value > (best_metrics[metric] + min_delta)
            for metric, value in metrics.items()
        }
        
        if loss_improved:
            best_loss = train_loss
            no_improve_loss = 0
        else:
            no_improve_loss += 1
        
        # 更新每个指标的改善状态
        for metric in metrics:
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
                'best_metrics': best_metrics,
                'vocab_size': vocab_size,
                'vocab_idx2word': vocab_idx2word,
                'initial_lr': initial_lr
            }, f'{checkpoint_path}/best_model.pth')
            print('\nNew best model saved!')
            print('Best metrics:')
            for metric, value in best_metrics.items():
                print(f'{metric}: {value:.4f}')
        
        # 打印改善状态
        print('\nNo improvement count:')
        print(f'Loss: {no_improve_loss}')
        for metric, count in no_improve_metrics.items():
            print(f'{metric}: {count}')
        
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
                print(f'- {metric} not improved for {count} epochs')
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
                'best_metrics': best_metrics,
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