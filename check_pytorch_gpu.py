import torch
import platform
import sys

def check_gpu():
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"操作系统: {platform.system()} {platform.release()}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA是否可用: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"检测到的GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} 内存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
        # 创建一个简单的张量并测试在GPU上运行
        print("\n正在进行简单的GPU测试...")
        try:
            # 创建随机矩阵
            a = torch.randn(1000, 1000).cuda()
            b = torch.randn(1000, 1000).cuda()
            
            # 计时矩阵乘法
            import time
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()  # 等待GPU操作完成
            end = time.time()
            
            print(f"矩阵乘法用时: {(end - start) * 1000:.2f} ms")
            print("GPU测试成功!")
        except Exception as e:
            print(f"GPU测试失败: {e}")
    else:
        print("未检测到GPU，将使用CPU进行训练（速度可能非常慢）")
        
        # 尝试诊断CUDA/GPU问题
        if platform.system() == "Windows":
            import subprocess
            import os
            
            # 检查NVIDIA驱动是否已安装
            try:
                nvidia_smi = subprocess.check_output("nvidia-smi", shell=True)
                print("\nNVIDIA驱动已安装:")
                print(nvidia_smi.decode('utf-8').split('\n')[0])
            except:
                print("\nNVIDIA驱动可能未安装或有问题")
                print("解决方法: 下载并安装最新的NVIDIA驱动")
            
            # 检查CUDA环境变量
            cuda_path = os.environ.get('CUDA_PATH')
            if cuda_path:
                print(f"\n检测到CUDA路径: {cuda_path}")
            else:
                print("\n未找到CUDA_PATH环境变量")
                print("解决方法: 安装CUDA工具包并设置环境变量")

if __name__ == "__main__":
    check_gpu() 