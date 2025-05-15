@echo off
echo 使用Anaconda运行PyTorch版本的CycleGAN训练...

REM 激活Anaconda环境
call conda activate pytorch_gan
if %ERRORLEVEL% neq 0 (
    echo 错误：无法激活pytorch_gan环境
    echo 请先按照setup_pytorch_anaconda.md中的说明创建环境
    exit /b 1
)

REM 查看GPU是否可用
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available()); print('检测到的GPU设备数量:', torch.cuda.device_count()); print('GPU设备名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU')"

REM 创建所需目录
if not exist "checkpoints" mkdir checkpoints
if not exist "generated_images" mkdir generated_images

REM 提示用户选择训练轮数
set /p EPOCHS="请输入训练轮数 (推荐: 5-20): "
if "%EPOCHS%"=="" set EPOCHS=5

REM 提示用户选择批次大小
set /p BATCH_SIZE="请输入批次大小 (推荐: 1-4，GPU内存较大可设置更高): "
if "%BATCH_SIZE%"=="" set BATCH_SIZE=2

REM 训练模型
echo 开始训练，使用 %EPOCHS% 轮，批次大小 %BATCH_SIZE%...
python monet_gan_pytorch.py --mode train --epochs %EPOCHS% --batch_size %BATCH_SIZE%

REM 提示用户是否生成提交文件
set /p GENERATE_SUBMISSION="训练完成，是否生成提交文件? (y/n): "
if /i "%GENERATE_SUBMISSION%"=="y" (
    echo 正在生成提交文件...
    python monet_gan_pytorch.py --mode predict
    echo 提交文件已生成: submission.zip
) else (
    echo 已跳过生成提交文件
)

REM 释放Anaconda环境
call conda deactivate

echo 训练过程完成！
pause 