# PyTorch版莫奈风格CycleGAN

这是一个使用PyTorch实现的CycleGAN模型，用于将普通照片转换为莫奈风格的图像。这个实现替代了原有的TensorFlow版本，提供更好的GPU兼容性和性能。

## 环境要求

- Python 3.6+
- PyTorch 1.8.0+
- CUDA (可选，但强烈推荐用于加速训练)

## 安装依赖

```bash
pip install -r requirements_pytorch.txt
```

## 数据结构

项目需要以下数据目录结构：

```
|-- monet_jpg/       # 莫奈画作图像
|-- photo_jpg/       # 真实照片
|-- checkpoints/     # 模型检查点保存路径（自动创建）
|-- generated_images/ # 生成的图像保存路径（自动创建）
```

## 使用方法

### 在Windows上训练

直接运行批处理文件：

```
run_pytorch_training.bat
```

或者手动运行Python命令：

```
python monet_gan_pytorch.py --mode train --epochs 5 --batch_size 2
```

### 参数说明

- `--mode`: 运行模式，可选 'train' 或 'predict'
- `--epochs`: 训练轮数
- `--batch_size`: 批处理大小（根据GPU内存调整）

## 生成提交文件

训练完成后，可以生成提交文件：

```
python monet_gan_pytorch.py --mode predict
```

这将在项目根目录下创建 `submission.zip` 文件，可以直接上传到Kaggle比赛页面。

## 模型架构

这个实现使用了标准的CycleGAN架构：

1. **生成器网络**：使用包含9个残差块的ResNet架构
2. **鉴别器网络**：使用PatchGAN架构
3. **损失函数**：
   - 对抗损失 (MSE)
   - 循环一致性损失 (L1)
   - 身份映射损失 (L1)

## 优化策略

- 使用Adam优化器
- 学习率衰减策略
- 图像缓冲区减少训练不稳定性
- 使用Instance Normalization提高生成图像质量

## 比较与TensorFlow版本的区别

- 使用PyTorch DataLoader提高数据加载效率
- 实现图像缓冲区来稳定训练
- 使用tqdm显示训练进度
- 更简洁的模型架构实现
- 更好的GPU兼容性和内存管理 