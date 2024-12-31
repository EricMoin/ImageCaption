import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import json
from models import ImageCaptioningModel
from transformers import GPT2Tokenizer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

def load_model(checkpoint_path):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 初始化tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'pad_token': '[PAD]',
        'bos_token': '[CLS]',
        'eos_token': '[SEP]',
        'unk_token': '[UNK]'
    }
    tokenizer.add_special_tokens(special_tokens)
    
    # 创建模型
    model = ImageCaptioningModel(vocab_size=len(tokenizer)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, device

def generate_description(model, image_path, tokenizer, device):
    """为单张图像生成描述"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 生成描述
    with torch.no_grad():
        output_ids = model.generate(
            images=image_tensor,
            tokenizer=tokenizer,
            max_length=100,
            min_length=10,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        
        # 解码生成的token
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text

def evaluate_metrics(generated_captions, reference_captions):
    """评估生成的描述"""
    # 准备评估器
    cider_scorer = Cider()
    spice_scorer = Spice()
    
    # 准备数据格式
    gts = {}  # 参考描述
    res = {}  # 生成的描述
    
    for i, (gen_cap, ref_caps) in enumerate(zip(generated_captions, reference_captions)):
        gts[i] = ref_caps
        res[i] = [gen_cap]
    
    # 计算分数
    cider_score, cider_scores = cider_scorer.compute_score(gts, res)
    spice_score, spice_scores = spice_scorer.compute_score(gts, res)
    
    return {
        'CIDEr-D': cider_score,
        'SPICE': spice_score,
        'CIDEr-D_per_image': cider_scores,
        'SPICE_per_image': spice_scores
    }

def main():
    # 配置参数
    checkpoint_path = 'checkpoints/best_model.pth'
    test_dir = 'data/test'
    reference_file = 'data/test_captions.json'  # 参考描述文件
    
    # 确保测试目录存在
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created test directory at {test_dir}")
        print("Please put test images in this directory.")
        return
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(test_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {test_dir}")
        print("Please add some images and run again.")
        return
    
    # 加载参考描述
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            references = json.load(f)
    except Exception as e:
        print(f"Error loading reference captions: {str(e)}")
        references = {}
    
    # 加载模型和tokenizer
    print("Loading model...")
    try:
        model, tokenizer, device = load_model(checkpoint_path)
        print(f"Using device: {device}")
        print("Model loaded successfully!")
        print(f"Vocabulary size: {len(tokenizer)}")
        
        print(f"\nProcessing {len(image_files)} images from {test_dir}...")
        
        # 存储生成的描述和参考描述
        generated_captions = []
        reference_captions = []
        
        # 处理每张图片
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(test_dir, image_file)
            try:
                # 生成描述
                description = generate_description(model, image_path, tokenizer, device)
                print(f"\nImage {i}/{len(image_files)}: {image_file}")
                print(f"Generated: {description}")
                
                # 获取参考描述
                if image_file in references:
                    ref_caps = references[image_file]
                    if isinstance(ref_caps, str):
                        ref_caps = [ref_caps]
                    elif isinstance(ref_caps, dict):
                        ref_caps = [ref_caps['caption']]
                    print(f"Reference: {ref_caps[0]}")
                    
                    # 存储描述用于评估
                    generated_captions.append(description)
                    reference_captions.append(ref_caps)
                
            except Exception as e:
                print(f"\nError processing image {image_file}: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        # 如果有参考描述，计算评估指标
        if generated_captions and reference_captions:
            print("\nCalculating evaluation metrics...")
            metrics = evaluate_metrics(generated_captions, reference_captions)
            print("\nEvaluation Results:")
            print(f"CIDEr-D Score: {metrics['CIDEr-D']:.4f}")
            print(f"SPICE Score: {metrics['SPICE']:.4f}")
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure the model checkpoint exists and is compatible.")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 