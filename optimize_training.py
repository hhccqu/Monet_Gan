import tensorflow as tf
import os
import multiprocessing

# 检查TensorFlow版本和是否支持GPU
print("TensorFlow版本:", tf.__version__)
print("TensorFlow编译信息:")
for key, value in tf.sysconfig.get_build_info().items():
    print(f"  {key}: {value}")

# 检查CPU核心数
cpu_count = multiprocessing.cpu_count()
print(f"\nCPU核心数: {cpu_count}")

# 配置CPU并行
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
print("已配置CPU并行线程")

# 修改monet_gan.py中的参数以优化训练
print("\n修改训练参数以优化性能:")

# 增加批处理大小
batch_size = 2  # 增加批量大小以提高CPU利用率
print(f"建议的批处理大小: {batch_size}")

# 减少epochs
epochs = 5  # 对于测试，减少epoch数量
print(f"建议的epochs数量: {epochs}")

print("\n使用这些参数运行训练可以提高CPU利用率。")
print("建议修改monet_gan.py文件中的以下参数:")
print("1. 将BATCH_SIZE从1改为", batch_size)
print("2. 将EPOCHS从10改为", epochs, "进行测试")
print("\n是否要自动修改这些参数？(y/n)")
choice = input()

if choice.lower() == 'y':
    # 读取文件内容
    with open('monet_gan.py', 'r') as file:
        content = file.read()
    
    # 替换批处理大小
    content = content.replace('BATCH_SIZE = 1', f'BATCH_SIZE = {batch_size}')
    
    # 替换epochs数量
    content = content.replace('EPOCHS = 10', f'EPOCHS = {epochs}')
    
    # 写回文件
    with open('monet_gan.py', 'w') as file:
        file.write(content)
    
    print("\n参数已更新。现在可以运行:")
    print("python monet_gan.py")
else:
    print("\n未修改参数。请手动修改后运行。") 