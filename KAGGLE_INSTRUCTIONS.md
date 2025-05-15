# 如何在Kaggle上运行Monet CycleGAN项目

本文档介绍如何在Kaggle平台上使用GitHub仓库中的代码运行莫奈风格转换CycleGAN项目。

## 步骤1：创建新的Kaggle Notebook

1. 登录到您的Kaggle账号
2. 访问"GAN Getting Started"竞赛页面：https://www.kaggle.com/c/gan-getting-started
3. 点击"Code"标签
4. 点击"New Notebook"按钮创建新的Notebook
5. 在设置中，选择"Accelerator" -> "GPU"（这一步很重要，因为GAN训练需要GPU）

## 步骤2：在Notebook中克隆GitHub仓库

将以下代码粘贴到Notebook的第一个代码单元中并运行：

```python
# 克隆GitHub仓库
!git clone https://github.com/hhccqu/Monet_Gan.git
!ls -la Monet_Gan
```

## 步骤3：设置数据路径

在新的代码单元中添加以下代码来设置正确的数据路径：

```python
import os
import glob

# 定义数据路径
MONET_DIR = "/kaggle/input/gan-getting-started/monet_jpg/"
PHOTO_DIR = "/kaggle/input/gan-getting-started/photo_jpg/"
OUTPUT_DIR = "/kaggle/working/generated_images/"
CHECKPOINT_DIR = "/kaggle/working/checkpoints/"

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 检查数据集
print(f"莫奈画作数量: {len(glob.glob(os.path.join(MONET_DIR, '*.jpg')))}")
print(f"照片数量: {len(glob.glob(os.path.join(PHOTO_DIR, '*.jpg')))}")
```

## 步骤4：修改CycleGAN代码以适应Kaggle环境

添加以下代码，修改路径以适应Kaggle环境：

```python
# 将代码复制到当前工作目录并修改路径
!cp Monet_Gan/monet_gan_pytorch.py .

# 修改文件中的路径
with open('monet_gan_pytorch.py', 'r') as f:
    code = f.read()

# 替换路径
code = code.replace('MONET_DIR = "monet_jpg/"', f'MONET_DIR = "{MONET_DIR}"')
code = code.replace('PHOTO_DIR = "photo_jpg/"', f'PHOTO_DIR = "{PHOTO_DIR}"')
code = code.replace('os.makedirs("checkpoints", exist_ok=True)', f'os.makedirs("{CHECKPOINT_DIR}", exist_ok=True)')
code = code.replace('f"checkpoints/checkpoint_{epoch}.pth"', f'"{CHECKPOINT_DIR}/checkpoint_{{epoch}}.pth"')
code = code.replace('os.makedirs("generated_images", exist_ok=True)', f'os.makedirs("{OUTPUT_DIR}", exist_ok=True)')
code = code.replace('f"generated_images/epoch{epoch}_batch{batch}_{i}.png"', f'"{OUTPUT_DIR}/epoch{{epoch}}_batch{{batch}}_{{i}}.png"')
code = code.replace('output_dir="generated_images"', f'output_dir="{OUTPUT_DIR}"')

# 保存修改后的文件
with open('monet_gan_pytorch_kaggle.py', 'w') as f:
    f.write(code)

print("代码已修改以适应Kaggle环境")
```

## 步骤5：安装必要的依赖并检查环境

```python
# 检查PyTorch版本和GPU可用性
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    
# 安装其他必要的依赖
!pip install tqdm
```

## 步骤6：导入修改后的模块并运行训练

对于快速测试，可以只运行1轮训练：

```python
# 导入修改后的CycleGAN模块
from monet_gan_pytorch_kaggle import train, predict_all_photos, prepare_submission

# 运行1轮训练作为演示
train(num_epochs=1)
```

对于完整训练（需要更长时间），建议运行5-10轮或更多：

```python
# 运行完整训练
train(num_epochs=5)
```

## 步骤7：生成预测结果

```python
# 生成预测并准备提交
predict_all_photos(output_dir=OUTPUT_DIR)
prepare_submission()
```

## 步骤8：可视化一些结果

```python
import matplotlib.pyplot as plt
from PIL import Image

def show_results(num_images=5):
    photo_paths = sorted(glob.glob(os.path.join(PHOTO_DIR, "*.jpg")))[:num_images]
    fig, axes = plt.subplots(num_images, 2, figsize=(12, 3*num_images))
    
    for i, photo_path in enumerate(photo_paths):
        # 加载原图
        photo = Image.open(photo_path).resize((256, 256))
        
        # 加载生成图
        filename = os.path.basename(photo_path)
        gen_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(gen_path):
            gen_img = Image.open(gen_path)
        else:
            gen_img = Image.new('RGB', (256, 256), color='gray')
        
        # 显示图像
        axes[i, 0].imshow(photo)
        axes[i, 0].set_title("原始照片")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(gen_img)
        axes[i, 1].set_title("莫奈风格")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()

show_results(5)
```

## 步骤9：保存和提交结果

生成的submission.zip文件将位于"/kaggle/working/submission.zip"，可以直接用于比赛提交：

```python
from IPython.display import FileLink

if os.path.exists('/kaggle/working/submission.zip'):
    print("提交文件已生成，可以直接提交到Kaggle竞赛")
    display(FileLink('/kaggle/working/submission.zip'))
else:
    print("提交文件未找到，请检查是否有错误发生")
```

## 注意事项

1. **GPU使用**：确保选择GPU加速器，否则训练会非常慢
2. **训练时间**：每轮训练可能需要20-30分钟，具体取决于GPU性能
3. **Kaggle时限**：Kaggle Notebook的运行时间有限制，对于完整训练可能需要保存检查点并分多次运行
4. **提交频率**：Kaggle对提交次数有限制，请合理安排测试和正式提交

祝您在Kaggle比赛中取得好成绩！ 