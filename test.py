import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from train import generate_caption  # 导入generate_caption函数

def load_image(image_path):
    """加载并预处理图像"""
    try:
        # 定义图像转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image)
        return image_tensor.unsqueeze(0)  # 添加batch维度
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def generate_description(model, image_path, vocab_idx2word, device):
    """为图像生成描述"""
    model.eval()  # 设置为评估模式
    
    # 加载和预处理图像
    image_tensor = load_image(image_path)
    if image_tensor is None:
        return "Error: Could not process image."
    
    image_tensor = image_tensor.to(device)
    
    # 生成描述
    with torch.no_grad():  # 不计算梯度
        try:
            # 获取句号的token ID
            PERIOD_TOKEN = None
            for idx, word in vocab_idx2word.items():
                if word == '.':
                    PERIOD_TOKEN = idx
                    break
            
            if PERIOD_TOKEN is None:
                print("Warning: Period token not found in vocabulary")
                return "Error: Period token not found in vocabulary."
            
            generated_ids = generate_caption(model, image_tensor, device, vocab_idx2word)
            if generated_ids is None:
                return "Error: Could not generate caption."
            
            # 将token ID转换为文本
            sentences = []
            current_sentence = []
            
            for idx in generated_ids[0]:
                token = vocab_idx2word.get(idx.item())
                if token and token not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    if token == '.':
                        if current_sentence:  # 只有当当前句子不为空时才添加句号
                            current_sentence.append(token)
                            sentences.append(' '.join(current_sentence))
                            current_sentence = []
                    else:
                        current_sentence.append(token)
            
            # 处理最后一个未完成的句子
            if current_sentence:
                current_sentence.append('.')
                sentences.append(' '.join(current_sentence))
            
            # 组合所有句子
            text = ' '.join(sentences)
            
            # 清理文本
            text = ' '.join(text.split())  # 移除多余的空格
            
            return text
            
        except Exception as e:
            print(f"Error generating description: {e}")
            return "Error: Could not generate description."

def main():
    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # 加载模型检查点
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
        
        # 获取词表大小和词表映射
        vocab_size = checkpoint['vocab_size']
        vocab_idx2word = checkpoint['vocab_idx2word']
        
        # 确保词表索引是整数类型
        vocab_idx2word = {int(idx): word for idx, word in vocab_idx2word.items()}
        
        # 打印词表信息
        print(f"Vocabulary size: {len(vocab_idx2word)}")
        print("Special tokens:", [word for idx, word in vocab_idx2word.items() if idx < 5])
        print("Punctuation marks:", [word for word in vocab_idx2word.values() if word in ['.', ',', '!', '?', ';', ':']])
        
        # 重新创建模型
        from models import ImageCaptioningModel
        model = ImageCaptioningModel(vocab_size=vocab_size).to(device)
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
        
        # 设置为评估模式
        model.eval()
        
        # 处理测试文件夹中的所有图像
        test_dir = 'data/test'
        if not os.path.exists(test_dir):
            print(f"Test directory {test_dir} does not exist!")
            return
        
        print("\nGenerating descriptions for test images...")
        for image_name in os.listdir(test_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(test_dir, image_name)
                try:
                    description = generate_description(model, image_path, vocab_idx2word, device)
                    print(f"\nImage: {image_name}")
                    print(f"Description: {description}")
                except Exception as e:
                    print(f"Error processing image {image_name}: {e}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

if __name__ == '__main__':
    main() 