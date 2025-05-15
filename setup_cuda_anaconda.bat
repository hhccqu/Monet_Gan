@echo off
echo 设置CUDA环境变量...

REM 检测CUDA路径
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6
if not exist "%CUDA_PATH%" (
    echo 未找到CUDA v11.6，尝试检测其他版本...
    
    for /d %%i in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do (
        if exist "%%i" (
            set CUDA_PATH=%%i
            echo 已找到CUDA: !CUDA_PATH!
            goto :found_cuda
        )
    )
    
    echo 错误：未找到CUDA安装。请安装CUDA工具包。
    echo 您可以从https://developer.nvidia.com/cuda-downloads下载
    exit /b 1
)

:found_cuda
echo 使用CUDA路径: %CUDA_PATH%

REM 设置环境变量
set PATH=%CUDA_PATH%\bin;%PATH%
set PATH=%CUDA_PATH%\libnvvp;%PATH%
set PATH=%CUDA_PATH%\extras\CUPTI\lib64;%PATH%

REM 检查NVIDIA驱动
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo 警告: 未找到NVIDIA驱动或无法运行nvidia-smi
    echo 请安装最新的NVIDIA驱动以使用GPU
) else (
    echo NVIDIA驱动检测正常
    nvidia-smi
)

REM 激活Anaconda环境
call conda activate pytorch_gan
if %ERRORLEVEL% neq 0 (
    echo 错误：无法激活pytorch_gan环境
    echo 请先按照setup_pytorch_anaconda.md中的说明创建环境
    exit /b 1
)

REM 运行PyTorch GPU检查
python check_pytorch_gpu.py

echo 环境变量设置完成！
echo 现在您可以运行run_conda_training.bat来训练模型

pause 