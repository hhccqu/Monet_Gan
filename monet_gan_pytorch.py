import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import random
import time
from tqdm import tqdm

# 设置随机种子以保证可复现性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义图像大小和批处理大小
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 2

# 数据路径
MONET_DIR = "monet_jpg/"
PHOTO_DIR = "photo_jpg/"

# 自定义数据集
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到 [-1, 1]
])

# 创建数据集和数据加载器
def create_dataloader(image_dir, batch_size):
    dataset = ImageDataset(image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader

# 生成器网络的残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

# 生成器网络
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=9):
        super(Generator, self).__init__()
        
        # 初始卷积层
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # 下采样
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # 残差块
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # 上采样
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # 输出层
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# 鉴别器网络
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        # PatchGAN鉴别器
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化模型权重
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 初始化CycleGAN模型
# 从照片到莫奈的生成器和鉴别器
netG_P2M = Generator().to(device)
netD_M = Discriminator().to(device)

# 从莫奈到照片的生成器和鉴别器
netG_M2P = Generator().to(device)
netD_P = Discriminator().to(device)

# 应用权重初始化
netG_P2M.apply(weights_init_normal)
netD_M.apply(weights_init_normal)
netG_M2P.apply(weights_init_normal)
netD_P.apply(weights_init_normal)

# 定义损失函数
criterion_GAN = nn.MSELoss()  # 对抗损失
criterion_cycle = nn.L1Loss()  # 循环一致性损失
criterion_identity = nn.L1Loss()  # 身份损失

# 定义优化器
optimizer_G = optim.Adam(
    list(netG_P2M.parameters()) + list(netG_M2P.parameters()),
    lr=0.0002,
    betas=(0.5, 0.999)
)
optimizer_D_M = optim.Adam(netD_M.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_P = optim.Adam(netD_P.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 学习率调度器
lr_scheduler_G = optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 50) / 50
)
lr_scheduler_D_M = optim.lr_scheduler.LambdaLR(
    optimizer_D_M, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 50) / 50
)
lr_scheduler_D_P = optim.lr_scheduler.LambdaLR(
    optimizer_D_P, lr_lambda=lambda epoch: 1.0 - max(0, epoch - 50) / 50
)

# 设置损失权重
lambda_cyc = 10.0  # 循环一致性损失权重
lambda_id = 5.0    # 身份损失权重

# 重置梯度
def reset_grad():
    optimizer_G.zero_grad()
    optimizer_D_M.zero_grad()
    optimizer_D_P.zero_grad()

# 创建图像缓冲区
class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        results = []
        for element in data.detach():
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                results.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    temp = self.data[i].clone()
                    self.data[i] = element
                    results.append(temp)
                else:
                    results.append(element)
        return torch.cat(results)

fake_monet_buffer = ImageBuffer()
fake_photo_buffer = ImageBuffer()

# 创建检查点保存目录
os.makedirs("checkpoints", exist_ok=True)

# 保存模型函数
def save_checkpoint(epoch):
    torch.save({
        'epoch': epoch,
        'netG_P2M_state_dict': netG_P2M.state_dict(),
        'netG_M2P_state_dict': netG_M2P.state_dict(),
        'netD_M_state_dict': netD_M.state_dict(),
        'netD_P_state_dict': netD_P.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_M_state_dict': optimizer_D_M.state_dict(),
        'optimizer_D_P_state_dict': optimizer_D_P.state_dict(),
    }, f"checkpoints/checkpoint_{epoch}.pth")
    print(f"模型已保存: checkpoint_{epoch}.pth")

# 加载模型函数
def load_checkpoint(checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        netG_P2M.load_state_dict(checkpoint['netG_P2M_state_dict'])
        netG_M2P.load_state_dict(checkpoint['netG_M2P_state_dict'])
        netD_M.load_state_dict(checkpoint['netD_M_state_dict'])
        netD_P.load_state_dict(checkpoint['netD_P_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D_M.load_state_dict(checkpoint['optimizer_D_M_state_dict'])
        optimizer_D_P.load_state_dict(checkpoint['optimizer_D_P_state_dict'])
        print(f"已加载检查点: {checkpoint_path}，从第 {epoch} 轮开始训练")
        return epoch
    else:
        print(f"未找到检查点: {checkpoint_path}，从头开始训练")
        return 0

# 生成图像函数
def generate_images(real_photo, epoch, batch):
    netG_P2M.eval()
    with torch.no_grad():
        fake_monet = netG_P2M(real_photo)
    
    # 转换为numpy数组进行可视化
    real_photo_np = real_photo.cpu().detach().numpy()
    fake_monet_np = fake_monet.cpu().detach().numpy()
    
    # 将图像从 [-1, 1] 转换回 [0, 1]
    real_photo_np = (real_photo_np + 1) / 2.0
    fake_monet_np = (fake_monet_np + 1) / 2.0
    
    # 创建图像目录
    os.makedirs("generated_images", exist_ok=True)
    
    # 保存生成的图像
    for i in range(fake_monet_np.shape[0]):
        # 转置通道顺序从 (C,H,W) 到 (H,W,C)
        real = np.transpose(real_photo_np[i], (1, 2, 0))
        fake = np.transpose(fake_monet_np[i], (1, 2, 0))
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(real)
        axs[0].set_title("真实照片")
        axs[0].axis("off")
        
        axs[1].imshow(fake)
        axs[1].set_title("莫奈风格")
        axs[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(f"generated_images/epoch{epoch}_batch{batch}_{i}.png")
        plt.close()
    
    netG_P2M.train()

# 训练函数
def train(num_epochs=5):
    # 创建数据加载器
    monet_dataloader = create_dataloader(MONET_DIR, BATCH_SIZE)
    photo_dataloader = create_dataloader(PHOTO_DIR, BATCH_SIZE)
    
    # 检查是否有检查点文件
    start_epoch = 0
    latest_checkpoint = None
    checkpoint_files = sorted(glob.glob("checkpoints/checkpoint_*.pth"))
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        start_epoch = load_checkpoint(latest_checkpoint)
    
    print(f"开始训练，总共 {num_epochs} 轮")
    
    for epoch in range(start_epoch, num_epochs):
        # 训练循环
        epoch_start_time = time.time()
        
        # 确保迭代器能够重复使用
        monet_iter = iter(monet_dataloader)
        photo_iter = iter(photo_dataloader)
        
        n_batches = min(len(monet_dataloader), len(photo_dataloader))
        
        # 使用tqdm显示进度
        with tqdm(total=n_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i in range(n_batches):
                # 获取下一批次的数据
                try:
                    real_monet = next(monet_iter).to(device)
                except StopIteration:
                    monet_iter = iter(monet_dataloader)
                    real_monet = next(monet_iter).to(device)
                
                try:
                    real_photo = next(photo_iter).to(device)
                except StopIteration:
                    photo_iter = iter(photo_dataloader)
                    real_photo = next(photo_iter).to(device)
                
                # 真实图像的标签
                real_label = torch.ones(real_monet.size(0), 1, 16, 16).to(device)
                fake_label = torch.zeros(real_monet.size(0), 1, 16, 16).to(device)
                
                ###################################
                # 训练生成器
                ###################################
                reset_grad()
                
                # 身份损失
                # G_M2P(monet) 应该等于 monet
                same_photo = netG_M2P(real_photo)
                loss_identity_photo = criterion_identity(same_photo, real_photo) * lambda_id
                
                # G_P2M(photo) 应该等于 photo
                same_monet = netG_P2M(real_monet)
                loss_identity_monet = criterion_identity(same_monet, real_monet) * lambda_id
                
                # GAN 损失
                fake_monet = netG_P2M(real_photo)
                pred_fake = netD_M(fake_monet)
                loss_GAN_P2M = criterion_GAN(pred_fake, real_label)
                
                fake_photo = netG_M2P(real_monet)
                pred_fake = netD_P(fake_photo)
                loss_GAN_M2P = criterion_GAN(pred_fake, real_label)
                
                # 循环一致性损失
                recovered_photo = netG_M2P(fake_monet)
                loss_cycle_P2M2P = criterion_cycle(recovered_photo, real_photo) * lambda_cyc
                
                recovered_monet = netG_P2M(fake_photo)
                loss_cycle_M2P2M = criterion_cycle(recovered_monet, real_monet) * lambda_cyc
                
                # 总生成器损失
                loss_G = (
                    loss_identity_photo + loss_identity_monet +
                    loss_GAN_P2M + loss_GAN_M2P +
                    loss_cycle_P2M2P + loss_cycle_M2P2M
                )
                
                # 反向传播
                loss_G.backward()
                optimizer_G.step()
                
                ###################################
                # 训练鉴别器 - Monet
                ###################################
                reset_grad()
                
                # 真实图像
                pred_real = netD_M(real_monet)
                loss_D_real = criterion_GAN(pred_real, real_label)
                
                # 假图像
                fake_monet_ = fake_monet_buffer.push_and_pop(fake_monet)
                pred_fake = netD_M(fake_monet_.detach())
                loss_D_fake = criterion_GAN(pred_fake, fake_label)
                
                # 总鉴别器Monet损失
                loss_D_M = (loss_D_real + loss_D_fake) * 0.5
                loss_D_M.backward()
                optimizer_D_M.step()
                
                ###################################
                # 训练鉴别器 - Photo
                ###################################
                reset_grad()
                
                # 真实图像
                pred_real = netD_P(real_photo)
                loss_D_real = criterion_GAN(pred_real, real_label)
                
                # 假图像
                fake_photo_ = fake_photo_buffer.push_and_pop(fake_photo)
                pred_fake = netD_P(fake_photo_.detach())
                loss_D_fake = criterion_GAN(pred_fake, fake_label)
                
                # 总鉴别器Photo损失
                loss_D_P = (loss_D_real + loss_D_fake) * 0.5
                loss_D_P.backward()
                optimizer_D_P.step()
                
                # 打印损失信息
                pbar.set_postfix({
                    'G_loss': f"{loss_G.item():.4f}",
                    'D_M_loss': f"{loss_D_M.item():.4f}",
                    'D_P_loss': f"{loss_D_P.item():.4f}"
                })
                pbar.update(1)
                
                # 每100个批次生成一些图像
                if i % 100 == 0:
                    generate_images(real_photo, epoch+1, i)
        
        # 更新学习率
        lr_scheduler_G.step()
        lr_scheduler_D_M.step()
        lr_scheduler_D_P.step()
        
        # 计算一轮训练的时间
        epoch_time = time.time() - epoch_start_time
        print(f"第 {epoch+1} 轮训练完成，耗时 {epoch_time:.2f} 秒")
        
        # 保存检查点
        save_checkpoint(epoch+1)
    
    print("训练完成!")

# 准备提交函数
def predict_all_photos(output_dir="generated_images"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建照片数据集（不要打乱顺序）
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    photo_dataset = ImageDataset(PHOTO_DIR, transform)
    photo_dataloader = DataLoader(photo_dataset, batch_size=1, shuffle=False)
    
    # 设置为评估模式
    netG_P2M.eval()
    
    print(f"开始生成所有照片的莫奈风格版本，总共 {len(photo_dataloader)} 张...")
    
    # 为每张照片生成对应的莫奈风格图像
    for i, real_photo in enumerate(tqdm(photo_dataloader)):
        real_photo = real_photo.to(device)
        
        with torch.no_grad():
            fake_monet = netG_P2M(real_photo)
        
        # 将结果转换回CPU，并从[-1,1]转换为[0,1]范围
        fake_monet = (fake_monet.cpu().detach() + 1) / 2.0
        
        # 获取原始文件名
        original_filename = os.path.basename(photo_dataset.image_paths[i])
        
        # 保存生成的图像
        generated_img = transforms.ToPILImage()(fake_monet[0])
        generated_img.save(f"{output_dir}/{original_filename}")
        
    print(f"所有图像已生成并保存到 {output_dir}!")

def prepare_submission():
    # 创建临时目录
    import tempfile
    import zipfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在生成所有测试图像的莫奈风格版本...")
    predict_all_photos(output_dir)
    
    # 创建提交文件
    submission_file = "submission.zip"
    
    print(f"正在创建提交文件 {submission_file}...")
    with zipfile.ZipFile(submission_file, 'w') as zip_file:
        for filename in os.listdir(output_dir):
            zip_file.write(os.path.join(output_dir, filename), filename)
    
    # 清理临时文件
    shutil.rmtree(temp_dir)
    
    print(f"提交文件已创建: {submission_file}")
    print("您可以将此文件上传到Kaggle比赛页面进行提交")

# 如果直接运行此脚本，开始训练
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用PyTorch训练CycleGAN生成莫奈风格图像")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"], help="运行模式: train或predict")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批处理大小")
    
    args = parser.parse_args()
    
    # 更新批处理大小
    BATCH_SIZE = args.batch_size
    
    if args.mode == "train":
        train(num_epochs=args.epochs)
    elif args.mode == "predict":
        # 检查模型是否已训练
        checkpoint_files = sorted(glob.glob("checkpoints/checkpoint_*.pth"))
        if checkpoint_files:
            load_checkpoint(checkpoint_files[-1])
            prepare_submission()
        else:
            print("错误: 未找到训练好的模型。请先运行训练模式!") 