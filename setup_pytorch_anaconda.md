# 使用Anaconda配置PyTorch CycleGAN环境

本指南将帮助您使用Anaconda配置PyTorch环境，并运行莫奈风格的CycleGAN图像生成模型。

## 1. 安装Anaconda

如果您尚未安装Anaconda，请从官方网站下载并安装：
https://www.anaconda.com/products/distribution

## 2. 创建PyTorch环境

打开Anaconda Prompt (开始菜单 -> Anaconda3 -> Anaconda Prompt)，然后运行以下命令：

```bash
# 创建名为pytorch_gan的新环境，使用Python 3.8
conda create -n pytorch_gan python=3.8

# 激活环境
conda activate pytorch_gan

# 安装PyTorch (GPU版本)
# 对于CUDA 11.6
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge

# 或者，如果您想安装仅CPU版本
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

## 3. 安装其他依赖包

```bash
# 安装必要的依赖
conda install numpy matplotlib pillow tqdm
conda install -c conda-forge glob2
```

## 4. 验证PyTorch安装和GPU可用性

```bash
# 在激活的环境中运行
python check_pytorch_gpu.py
```

这将显示您的PyTorch版本和GPU状态。确保CUDA是可用的，并且您的GPU被正确识别。

## 5. 运行训练

### 方法1: 使用批处理文件

```bash
# 确保当前在项目目录中
cd D:\KAGGLE\gan-getting-started

# 确保环境已激活
conda activate pytorch_gan

# 运行批处理文件
run_pytorch_training.bat
```

### 方法2: 直接使用Python命令

```bash
# 确保环境已激活
conda activate pytorch_gan

# 训练模型
python monet_gan_pytorch.py --mode train --epochs 5 --batch_size 2

# 生成提交文件
python monet_gan_pytorch.py --mode predict
```

## 6. 调整训练参数

您可以调整以下参数来优化训练：

- **批处理大小**：如果有足够的GPU内存，增加 `--batch_size` 可以加速训练
- **训练轮数**：增加 `--epochs` 可以提高生成图像的质量
- **GPU配置**：如果您有多个GPU，可以在 `monet_gan_pytorch.py` 中修改设备选择

## 7. 提交结果

训练完成后，将在项目目录中生成 `submission.zip` 文件，可以直接上传到Kaggle比赛页面。

## 8. 环境管理备注

- 使用 `conda activate pytorch_gan` 激活环境
- 使用 `conda deactivate` 退出环境
- 如果需要删除环境：`conda env remove -n pytorch_gan`

## 9. 故障排除

### 如果CUDA不可用：

1. 确认您的NVIDIA驱动是最新的
2. 确认安装了与您的GPU兼容的CUDA版本
3. 检查环境变量是否正确设置

### 如果遇到内存错误：

1. 减小批处理大小
2. 减小图像大小（在代码中修改 `IMG_HEIGHT` 和 `IMG_WIDTH`）
3. 使用梯度累积技术（需要修改代码） 