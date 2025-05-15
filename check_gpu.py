import tensorflow as tf
import os

print("TensorFlow版本:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("可用的GPU设备:", gpus)
print("GPU是否可用:", len(gpus) > 0)

# 打印CUDA环境变量
print("\nCUDA环境变量:")
for env_var in ['CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'CUDA_HOME']:
    print(f"{env_var}: {os.environ.get(env_var, '未设置')}")

# 尝试设置内存增长
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("\n成功设置GPU内存增长")
    except RuntimeError as e:
        print(f"\nGPU设置错误: {e}") 