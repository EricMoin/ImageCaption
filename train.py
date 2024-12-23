import torch
import torchvision
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import numpy as np
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device, tokenizer, max_len=50):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    print(f"Training on {len(dataloader.dataset)} samples in {num_batches} batches")
    
    # 梯度裁剪阈值
    grad_clip = 1.0
    
    for batch_idx, (images, captions) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        captions = captions.to(device)
        
        # 准备输入和标签
        input_ids = captions
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        
        # 将padding token的标签设为-100（忽略）
        labels[labels == tokenizer.pad_token_id] = -100
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs['loss']
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss

def evaluate_metrics(model, dataloader, device, tokenizer):
    model.eval()
    references = []
    hypotheses = []
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    print(f"\nEvaluating on {len(dataloader.dataset)} samples")
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            
            # 生成描述
            generated_ids = model.generate(
                images=images,
                tokenizer=tokenizer,
                max_length=50,
                num_beams=4,
                temperature=0.7
            )
            
            # 解码生成的描述和参考描述
            for gen_ids, cap_ids in zip(generated_ids, captions):
                # 处理生成的描述
                pred_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                pred_tokens = pred_text.split()
                if not pred_tokens:
                    pred_tokens = ['<unk>']
                hypotheses.append(pred_tokens)
                
                # 处理参考描述
                ref_text = tokenizer.decode(cap_ids, skip_special_tokens=True)
                ref_tokens = ref_text.split()
                if not ref_tokens:
                    ref_tokens = ['<unk>']
                references.append([ref_tokens])
    
    # 计算各种评估指标
    try:
        # BLEU分数
        bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))
        bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
        
        # METEOR分数
        meteor_scores = []
        for hyp, ref in zip(hypotheses, references):
            score = meteor_score(ref, hyp)
            meteor_scores.append(score)
        meteor = np.mean(meteor_scores)
        
        # ROUGE-L分数
        rouge_scores = []
        for hyp, ref in zip(hypotheses, references):
            score = rouge_scorer_obj.score(' '.join(hyp), ' '.join(ref[0]))
            rouge_scores.append(score['rougeL'].fmeasure)
        rouge_l = np.mean(rouge_scores)
        
        # 打印样本结果
        print("\nSample predictions:")
        for i in range(min(3, len(hypotheses))):
            print(f"\nReference: {' '.join(references[i][0])}")
            print(f"Generated: {' '.join(hypotheses[i])}")
            
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None
    
    return {
        'bleu1': bleu1,
        'bleu4': bleu4,
        'meteor': meteor,
        'rouge_l': rouge_l
    }

def train_model(model, train_loader, val_loader, num_epochs, optimizer, scheduler, 
                device, checkpoint_path, tokenizer):
    best_metrics = {
        'bleu1': 0,
        'bleu4': 0,
        'meteor': 0,
        'rouge_l': 0
    }
    best_loss = float('inf')
    patience = 5
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, tokenizer)
        print(f'\nAverage Training Loss: {train_loss:.4f}')
        
        # 评估
        metrics = evaluate_metrics(model, val_loader, device, tokenizer)
        if metrics is not None:
            print('\nValidation Metrics:')
            for metric, value in metrics.items():
                print(f'{metric.upper()}: {value:.4f}')
            
            # 更新学习率
            scheduler.step(metrics['bleu4'])
            
            # 检查是否有改善
            improved = False
            for metric, value in metrics.items():
                if value > best_metrics[metric]:
                    best_metrics[metric] = value
                    improved = True
            
            if train_loss < best_loss:
                best_loss = train_loss
                improved = True
            
            if improved:
                no_improve_epochs = 0
                # 保存模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': train_loss,
                    'metrics': metrics
                }, f'{checkpoint_path}/best_model.pth')
                print('\nNew best model saved!')
            else:
                no_improve_epochs += 1
                print(f'\nNo improvement for {no_improve_epochs} epochs')
            
            # 早停
            if no_improve_epochs >= patience:
                print(f'\nEarly stopping after {patience} epochs without improvement')
                break
        
        print('-' * 50)

def generate_description(model, image_path, transform, tokenizer, device):
    model.eval()
    
    # 加载和预处理图像
    image = torchvision.io.read_image(image_path)
    image = transform(image).unsqueeze(0).to(device)
    
    # 生成描述
    with torch.no_grad():
        generated_ids = model.generate(
            images=image,
            tokenizer=tokenizer,
            max_length=50,
            num_beams=4,
            temperature=0.7
        )
        
        # 解码生成的描述
        description = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return description 