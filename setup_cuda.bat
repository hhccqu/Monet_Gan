@echo off
echo 设置CUDA环境变量...

:: 设置CUDA路径
set "CUDA_PATH=D:\AirmodelingTeam\monet_gan\Library\bin"
set "PATH=%CUDA_PATH%;%PATH%"
set "CUDA_VISIBLE_DEVICES=0"

:: 设置重要的DLL所在目录
set "CUDNN_PATH=%CUDA_PATH%"
set "LD_LIBRARY_PATH=%CUDA_PATH%;%LD_LIBRARY_PATH%"

:: 显示设置的环境变量
echo CUDA环境变量已设置:
echo CUDA_PATH=%CUDA_PATH%
echo CUDNN_PATH=%CUDNN_PATH%
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%

:: 尝试修复TensorFlow找不到cudnn的问题
:: 这是一种临时解决方案，将cudnn64_8.dll重新注册到系统中
echo 尝试使TensorFlow能找到CUDA库...
cd %CUDA_PATH%

:: 检查GPU状态
echo 检查GPU状态...
python %~dp0check_gpu.py

:: 运行GAN训练脚本
echo 运行monet_gan.py...
cd %~dp0
python monet_gan.py 