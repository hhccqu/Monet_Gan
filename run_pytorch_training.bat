@echo off
echo 开始运行PyTorch版本的CycleGAN训练...

REM 查看GPU是否可用
python -c "import torch; print('CUDA是否可用:', torch.cuda.is_available()); print('检测到的GPU设备数量:', torch.cuda.device_count()); print('GPU设备名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无GPU')"

REM 创建所需目录
if not exist "checkpoints" mkdir checkpoints
if not exist "generated_images" mkdir generated_images

REM 训练模型
echo 使用GPU进行训练...
python monet_gan_pytorch.py --mode train --epochs 5 --batch_size 2

REM 生成提交结果
echo 正在生成提交文件...
python monet_gan_pytorch.py --mode predict

echo 训练和生成完成！
echo 提交文件已生成: submission.zip 