import os
import sys
import subprocess
import tensorflow as tf

# 设置CUDA环境变量
cuda_path = "D:\\AirmodelingTeam\\monet_gan\\Library\\bin"
os.environ["CUDA_PATH"] = cuda_path
os.environ["PATH"] = cuda_path + ";" + os.environ.get("PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = cuda_path

# 重要：强制TensorFlow使用GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# 增加日志级别以查看更多信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 显示所有日志

print("环境变量已设置:")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', '未设置')}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '未设置')}")
print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', '未设置')}")
print(f"TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', '未设置')}")

# 确认环境变量是否正确
print("\n检查DLL文件是否存在:")
cudnn_path = os.path.join(cuda_path, "cudnn64_8.dll")
print(f"cudnn64_8.dll 存在: {os.path.exists(cudnn_path)}")
cudart_path = os.path.join(cuda_path, "cudart64_110.dll")
print(f"cudart64_110.dll 存在: {os.path.exists(cudart_path)}")

# 打印TensorFlow和GPU信息
print("\nTensorFlow信息:")
print(f"TensorFlow版本: {tf.__version__}")
print(f"TensorFlow编译设置: {tf.sysconfig.get_build_info()}")
print("\nGPU信息:")
physical_devices = tf.config.list_physical_devices('GPU')
print(f"可用的GPU数量: {len(physical_devices)}")
for device in physical_devices:
    print(f"  {device}")

# 询问用户是否继续运行
print("\n似乎TensorFlow未能识别GPU。您是否希望仍然使用CPU运行训练？(y/n)")
choice = input()
if choice.lower() == 'y':
    print("\n开始使用CPU运行monet_gan.py...")
    subprocess.run([sys.executable, "monet_gan.py"])
else:
    print("已取消训练。")
    print("\n建议解决方案:")
    print("1. 确保已安装兼容的NVIDIA驱动程序")
    print("2. 尝试创建新的conda环境: conda create -n tf_gpu python=3.8 tensorflow-gpu=2.10.0")
    print("3. 参考TensorFlow官方文档: https://www.tensorflow.org/install/gpu") 