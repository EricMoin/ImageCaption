import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from models import ImageCaptioningModel
from train import generate_caption

def load_model(checkpoint_path):
    """加载模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    model = ImageCaptioningModel(vocab_size=checkpoint['vocab_size']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 确保词表索引是整数类型
    vocab_idx2word = {int(idx): word for idx, word in checkpoint['vocab_idx2word'].items()}
    
    return model, vocab_idx2word, device

def generate_description(model, image_path, vocab_idx2word, device):
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
        generated_ids = generate_caption(model, image_tensor, device,vocab_idx2word)
        try:
            tokens = [vocab_idx2word[idx.item()] for idx in generated_ids[0]
                     if idx.item() not in [0, 1, 2, 3]]  # 移除特殊token
        except KeyError as e:
            print(f"Warning: Unknown token ID: {e}")
            print("Available token IDs:", sorted(vocab_idx2word.keys()))
            tokens = []
    
    return ' '.join(tokens) if tokens else "Unable to generate description"

def main():
        # 配置参数
    checkpoint_path = 'checkpoints/best_model.pth'
    test_dir = 'data/test'
    
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
    
    # 加载模型和词表
    print("Loading model...")
    try:
        model, vocab_idx2word, device = load_model(checkpoint_path)
        print(f"Using device: {device}")
        print("Model loaded successfully!")
        print(f"Vocabulary size: {len(vocab_idx2word)}")
        print("Sample vocabulary items:", list(vocab_idx2word.items())[:5])
        
        print(f"\nProcessing {len(image_files)} images from {test_dir}...")
        
        # 处理每张图片
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(test_dir, image_file)
            try:
                description = generate_description(model, image_path, vocab_idx2word, device)
                print(f"\nImage {i}/{len(image_files)}: {image_file}")
                print(f"Description: {description}")
            except Exception as e:
                print(f"\nError processing image {image_file}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please make sure the model checkpoint exists and contains the vocabulary information.")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main() 