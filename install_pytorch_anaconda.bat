@echo off
echo 开始安装PyTorch Anaconda环境...

REM 检查conda命令是否可用
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 错误：未找到conda命令。请确保已安装Anaconda或Miniconda。
    echo 您可以从https://www.anaconda.com/products/distribution下载
    exit /b 1
)

echo 正在创建pytorch_gan环境...
call conda env create -f environment.yml

if %ERRORLEVEL% neq 0 (
    echo 环境创建失败，尝试更新现有环境...
    call conda env update -f environment.yml
    if %ERRORLEVEL% neq 0 (
        echo 环境更新失败。请查看错误信息并手动创建环境。
        exit /b 1
    )
)

echo 环境创建成功！

REM 激活环境并验证
call conda activate pytorch_gan

echo 验证PyTorch安装...
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA是否可用:', torch.cuda.is_available()); print('PyTorch安装路径:', torch.__file__)"

if %ERRORLEVEL% neq 0 (
    echo PyTorch验证失败，请检查安装
    exit /b 1
)

echo 安装完成！您现在可以使用以下命令来训练模型：
echo.
echo 1. 设置CUDA环境：setup_cuda_anaconda.bat
echo 2. 运行训练：run_conda_training.bat
echo.

pause 