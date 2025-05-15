# Anaconda环境设置指南

按照以下步骤设置Anaconda环境来运行莫奈风格转换GAN项目。

## 先决条件

- 已安装[Anaconda](https://www.anaconda.com/products/individual)或[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## 设置步骤

1. **克隆或下载项目代码**

2. **创建Anaconda环境**

   打开命令行终端（Anaconda Prompt或PowerShell），然后执行以下命令：

   ```bash
   # 导航到项目目录
   cd 路径/到/gan-getting-started

   # 使用environment.yml文件创建环境
   conda env create -f environment.yml
   ```

   这将创建一个名为`monet_gan`的新环境，并安装所有必需的依赖项。

3. **激活环境**

   ```bash
   conda activate monet_gan
   ```

4. **验证安装**

   ```bash
   # 验证TensorFlow安装
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

5. **运行项目**

   ```bash
   # 运行训练脚本
   python monet_gan.py
   ```

## 可能的GPU加速配置

如果您的系统有支持CUDA的NVIDIA GPU，可以考虑安装GPU版本的TensorFlow来加速训练。

1. **创建支持GPU的环境**（可选的替代方案）

   如果您有GPU并希望使用它进行训练，请使用以下命令创建环境：

   ```bash
   # 使用environment.yml创建基本环境
   conda env create -f environment.yml
   conda activate monet_gan
   
   # 卸载CPU版本的TensorFlow
   pip uninstall -y tensorflow tensorflow-addons
   
   # 安装GPU版本的TensorFlow（以及兼容的CUDA和cuDNN）
   conda install -c conda-forge tensorflow-gpu=2.4 cudatoolkit=11.0 cudnn=8.0
   pip install tensorflow-addons>=0.12.0
   ```

## 故障排除

- **内存错误**: 如果遇到内存不足的问题，尝试降低`monet_gan.py`中的`BATCH_SIZE`值。
- **GPU内存不足**: 如果使用GPU但出现内存错误，可能需要减小批次大小或图像大小。
- **依赖冲突**: 如果出现依赖项冲突，可以尝试创建一个全新的环境，并按照特定顺序手动安装依赖项。

## 环境管理命令

- 停用环境: `conda deactivate`
- 删除环境: `conda env remove -n monet_gan`
- 更新环境: `conda env update -f environment.yml` 