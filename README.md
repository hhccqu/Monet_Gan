# 莫奈风格转换 GAN

这个项目是 Kaggle 比赛 "I'm Something of a Painter Myself" 的解决方案，旨在使用生成对抗网络(GAN)将普通照片转换成莫奈画风的图像。

## 项目概述

本项目使用 CycleGAN 架构实现从普通照片到莫奈风格图像的转换。CycleGAN 的核心思想是通过循环一致性损失使模型能够在不成对的数据集之间学习域转换。

## 数据集

数据集已经下载到以下目录:
- `monet_jpg/`: 莫奈画作图像
- `photo_jpg/`: 普通照片图像
- `monet_tfrec/`和`photo_tfrec/`: TensorFlow记录格式的相同数据

## 环境设置

安装必要的依赖:

```bash
pip install -r requirements.txt
```

## 使用方法

1. **训练模型**:
   ```bash
   python monet_gan.py
   ```
   训练会自动开始，每隔5个epoch会保存检查点和生成样本图像。

2. **生成提交文件**:
   训练完成后，程序会自动生成`submission.zip`文件，包含所有转换后的图像。

## 模型架构

项目使用了两个生成器和两个鉴别器:
- Generator P2M: 将照片转换为莫奈风格
- Generator M2P: 将莫奈风格转换为照片
- Discriminator M: 区分真实的莫奈画作和生成的莫奈风格图像
- Discriminator P: 区分真实照片和生成的照片

## 训练过程

训练过程包含以下损失:
- 对抗损失: 使生成的图像骗过鉴别器
- 循环一致性损失: 确保经过双向转换后的图像与原始图像相似
- 身份损失: 保证当输入已经是目标域的图像时，不会产生过度变化

## 结果查看

训练过程中，每5个epoch会生成一张示例图像`generated_sample.png`，展示从照片到莫奈风格的转换效果。 