import os
import sys
import tensorflow as tf

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# 打印TensorFlow版本信息
print("TensorFlow版本:", tf.__version__)

# 检查编译信息
print("TensorFlow编译信息:")
for key, value in tf.sysconfig.get_build_info().items():
    print(f"  {key}: {value}")

# 检查GPU设备
gpus = tf.config.list_physical_devices('GPU')
print("\n可用的GPU设备:", gpus)
print("GPU是否可用:", len(gpus) > 0)

if len(gpus) > 0:
    # 尝试配置GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n成功配置GPU内存增长")
    except Exception as e:
        print(f"\nGPU配置错误: {e}")
    print("\n当前训练正在使用GPU")
else:
    print("\n当前训练未使用GPU，正在使用CPU进行训练")
    
    # 尝试输出更多诊断信息
    print("\n诊断信息:")
    print("1. 检查CUDA库是否存在:")
    cuda_path = "D:\\AirmodelingTeam\\monet_gan\\Library\\bin"
    for dll in ["cudart64_110.dll", "cudnn64_8.dll", "cublas64_11.dll"]:
        dll_path = os.path.join(cuda_path, dll)
        print(f"  {dll} 存在: {os.path.exists(dll_path)}")
        
    print("\n2. GPU支持状态:")
    print(f"  CUDA是否可用: {tf.test.is_built_with_cuda()}")
    print(f"  GPU是否可用: {tf.test.is_gpu_available()}")
    
    print("\n解决方案:")
    print("1. 重新安装GPU版本的TensorFlow: pip install tensorflow-gpu==2.10.0")
    print("2. 确保NVIDIA驱动程序已正确安装")
    print("3. 检查CUDA和cuDNN的版本是否与TensorFlow 2.10.0兼容") 