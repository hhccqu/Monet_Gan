import os
import sys
import subprocess
import platform

# 获取系统信息
print("Python版本:", platform.python_version())
print("操作系统:", platform.system(), platform.release())

print("\n尝试安装TensorFlow GPU版本...")

# 设置pip不使用代理
os.environ["PIP_NO_PROXY"] = "*"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# 尝试安装tensorflow-gpu
try:
    # 首先卸载旧版本的tensorflow
    print("卸载现有的TensorFlow...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"])
    
    # 使用pip直接安装tensorflow-gpu
    print("\n安装tensorflow-gpu...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow-gpu==2.10.0"], capture_output=True, text=True)
    
    # 输出安装结果
    print("\n安装输出:")
    print(result.stdout)
    
    if result.returncode != 0:
        print("\n安装错误:")
        print(result.stderr)
        
        # 尝试其他可用版本
        print("\n尝试安装最新版本的tensorflow...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow"], capture_output=True, text=True)
        
        print("\n安装输出:")
        print(result.stdout)
        
        if result.returncode != 0:
            print("\n安装错误:")
            print(result.stderr)
            print("\n无法安装TensorFlow，请尝试手动安装。")
        else:
            print("\nTensorFlow安装成功，现在验证是否支持GPU...")
    else:
        print("\nTensorFlow GPU安装成功，现在验证是否支持GPU...")
    
    # 验证安装
    import tensorflow as tf
    print("\nTensorFlow版本:", tf.__version__)
    print("GPU是否可用:", len(tf.config.list_physical_devices('GPU')) > 0)
    print("可用的GPU设备:", tf.config.list_physical_devices('GPU'))
    
except Exception as e:
    print(f"\n安装过程中出错: {e}")
    print("\n请尝试使用以下命令手动安装:")
    print("pip install tensorflow-gpu==2.10.0 --no-cache-dir")
    print("或者:")
    print("conda install tensorflow-gpu=2.10.0")

print("\n建议在安装完成后运行:")
print("python check_gpu_status.py")
print("来验证GPU是否可用。") 