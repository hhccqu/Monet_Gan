import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import multiprocessing

# 设置随机种子以保证可复现性
tf.random.set_seed(42)
np.random.seed(42)

# 配置GPU内存增长，避免一次性分配所有GPU内存
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        print(f"找到 {len(gpus)} 个GPU设备")
        for gpu in gpus:
            # 启用内存增长
            tf.config.experimental.set_memory_growth(gpu, True)
            # 设置内存限制（可选，根据实际情况设置）
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpu,
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        print("GPU内存增长已启用，将尝试使用GPU进行训练")
    except RuntimeError as e:
        print("GPU设置错误:", e)
    
# 检测可用的设备
print("TensorFlow版本:", tf.__version__)
print("可用的GPU设备:", gpus)

# 尝试使用GPU，如果有的话
if gpus:
    try:
        print("将使用GPU进行训练")
    except RuntimeError as e:
        print("GPU设置错误:", e)
else:
    print("没有检测到GPU，将使用CPU进行训练")

# 配置CPU并行
cpu_count = multiprocessing.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
tf.config.threading.set_inter_op_parallelism_threads(cpu_count) 
print(f"已配置CPU并行线程，使用 {cpu_count} 个核心")

# 定义图像大小
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 2  # 调整批处理大小以提高CPU利用率

# 数据路径
MONET_DIR = "monet_jpg/"
PHOTO_DIR = "photo_jpg/"

# 加载图像数据
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (image / 127.5) - 1  # 将像素值归一化到 [-1, 1]
    return image

# 创建数据集
def create_dataset(image_dir, batch_size):
    image_paths = glob(os.path.join(image_dir, "*.jpg"))
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

# 创建生成器网络 - Unet架构
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                             kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

# 生成器模型
def Generator():
    inputs = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    
    # 编码器
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]
    
    # 解码器
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)
    
    x = inputs
    
    # 下采样部分
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # 上采样部分与跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

# 创建鉴别器网络
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
    
    x = downsample(64, 4, False)(inp)  # (batch_size, 128, 128, 64)
    x = downsample(128, 4)(x)  # (batch_size, 64, 64, 128)
    x = downsample(256, 4)(x)  # (batch_size, 32, 32, 256)
    
    # 使用PatchGAN的方法
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(x)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
    
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)
    
    return tf.keras.Model(inputs=inp, outputs=last)

# 定义损失函数
def discriminator_loss(real, generated):
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(generated), generated)
    total_loss = real_loss + generated_loss
    return total_loss * 0.5

def generator_loss(generated):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated), generated)

# 定义CycleGAN的损失函数
def calc_cycle_loss(real_image, cycled_image):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return loss * 10.0  # 循环一致性损失权重

def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return loss * 0.5  # 身份损失权重

# 初始化CycleGAN模型
# 从照片到莫奈的生成器和鉴别器
generator_p2m = Generator()
discriminator_m = Discriminator()

# 从莫奈到照片的生成器和鉴别器
generator_m2p = Generator()
discriminator_p = Discriminator()

# 定义优化器
generator_p2m_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_m2p_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_m_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_p_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# 定义检查点
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(
    generator_p2m=generator_p2m,
    generator_m2p=generator_m2p,
    discriminator_m=discriminator_m,
    discriminator_p=discriminator_p,
    generator_p2m_optimizer=generator_p2m_optimizer,
    generator_m2p_optimizer=generator_m2p_optimizer,
    discriminator_m_optimizer=discriminator_m_optimizer,
    discriminator_p_optimizer=discriminator_p_optimizer
)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果有检查点恢复训练
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('最新的检查点已恢复')

# 训练步骤
@tf.function
def train_step(real_monet, real_photo):
    with tf.GradientTape(persistent=True) as tape:
        # 从照片生成莫奈风格画
        fake_monet = generator_p2m(real_photo, training=True)
        # 从莫奈画生成照片
        fake_photo = generator_m2p(real_monet, training=True)
        
        # 循环一致性 - 照片 -> 莫奈 -> 照片
        cycled_photo = generator_m2p(fake_monet, training=True)
        # 循环一致性 - 莫奈 -> 照片 -> 莫奈
        cycled_monet = generator_p2m(fake_photo, training=True)
        
        # 同一性映射 - 照片映射到照片
        same_photo = generator_m2p(real_photo, training=True)
        # 同一性映射 - 莫奈映射到莫奈
        same_monet = generator_p2m(real_monet, training=True)
        
        # 鉴别器对假样本的判别
        disc_fake_monet = discriminator_m(fake_monet, training=True)
        disc_fake_photo = discriminator_p(fake_photo, training=True)
        
        # 鉴别器对真样本的判别
        disc_real_monet = discriminator_m(real_monet, training=True)
        disc_real_photo = discriminator_p(real_photo, training=True)
        
        # 计算生成器损失
        gen_p2m_loss = generator_loss(disc_fake_monet)
        gen_m2p_loss = generator_loss(disc_fake_photo)
        
        # 计算循环一致性损失
        total_cycle_loss = calc_cycle_loss(real_monet, cycled_monet) + calc_cycle_loss(real_photo, cycled_photo)
        
        # 计算同一性损失
        total_identity_loss = identity_loss(real_monet, same_monet) + identity_loss(real_photo, same_photo)
        
        # 生成器总损失
        total_gen_p2m_loss = gen_p2m_loss + total_cycle_loss + total_identity_loss
        total_gen_m2p_loss = gen_m2p_loss + total_cycle_loss + total_identity_loss
        
        # 计算鉴别器损失
        disc_m_loss = discriminator_loss(disc_real_monet, disc_fake_monet)
        disc_p_loss = discriminator_loss(disc_real_photo, disc_fake_photo)
    
    # 计算梯度
    generator_p2m_gradients = tape.gradient(total_gen_p2m_loss, generator_p2m.trainable_variables)
    generator_m2p_gradients = tape.gradient(total_gen_m2p_loss, generator_m2p.trainable_variables)
    
    discriminator_m_gradients = tape.gradient(disc_m_loss, discriminator_m.trainable_variables)
    discriminator_p_gradients = tape.gradient(disc_p_loss, discriminator_p.trainable_variables)
    
    # 应用梯度
    generator_p2m_optimizer.apply_gradients(zip(generator_p2m_gradients, generator_p2m.trainable_variables))
    generator_m2p_optimizer.apply_gradients(zip(generator_m2p_gradients, generator_m2p.trainable_variables))
    
    discriminator_m_optimizer.apply_gradients(zip(discriminator_m_gradients, discriminator_m.trainable_variables))
    discriminator_p_optimizer.apply_gradients(zip(discriminator_p_gradients, discriminator_p.trainable_variables))
    
    return {
        "gen_p2m_loss": total_gen_p2m_loss,
        "gen_m2p_loss": total_gen_m2p_loss,
        "disc_m_loss": disc_m_loss,
        "disc_p_loss": disc_p_loss
    }

# 生成样本图像来监控训练进度
def generate_images(model, test_input):
    prediction = model(test_input)
    
    plt.figure(figsize=(12, 12))
    
    display_list = [test_input[0], prediction[0]]
    title = ['输入图像', '输出图像']
    
    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # 将图像从[-1, 1]转换回[0, 1]
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig('generated_sample.png')
    plt.close()

# 主训练循环
def train():
    # 创建数据集
    monet_dataset = create_dataset(MONET_DIR, BATCH_SIZE)
    photo_dataset = create_dataset(PHOTO_DIR, BATCH_SIZE)
    
    # 设置训练参数
    EPOCHS = 5  # 减少epoch数量加快训练速度
    
    # 保存用于生成图像的示例输入
    example_monet = next(iter(monet_dataset))
    example_photo = next(iter(photo_dataset))
    
    # 训练循环
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        
        # 将两个数据集合并为一个
        n = 0
        losses = {"gen_p2m_loss": 0, "gen_m2p_loss": 0, "disc_m_loss": 0, "disc_p_loss": 0}
        
        for monet_batch, photo_batch in tf.data.Dataset.zip((monet_dataset, photo_dataset)):
            batch_losses = train_step(monet_batch, photo_batch)
            
            # 累计损失
            for k in losses.keys():
                losses[k] += batch_losses[k]
            
            n += 1
            
            # 每10个批次打印一次进度
            if n % 10 == 0:
                progress = f"处理了 {n} 个批次"
                print(progress)
        
        # 显示平均损失
        losses = {k: v/n for k, v in losses.items()}
        print(f"Losses: {losses}")
        
        # 生成样本图片
        if (epoch + 1) % 1 == 0:  # 每个epoch都生成样本
            generate_images(generator_p2m, example_photo)
            
            # 保存检查点
            ckpt_save_path = ckpt_manager.save()
            print(f'保存检查点到 {ckpt_save_path}')

    # 训练完成后生成最终样本
    generate_images(generator_p2m, example_photo)

# 预测函数 - 用于将照片转换为莫奈风格
def predict(photo_path, output_path):
    # 加载图像
    image = load_image(photo_path)
    image = tf.expand_dims(image, 0)
    
    # 生成预测
    prediction = generator_p2m(image, training=False)
    
    # 将预测转换回[0, 1]范围
    prediction = (prediction * 0.5 + 0.5)
    
    # 保存预测结果
    pred_image = tf.cast(prediction[0] * 255, tf.uint8)
    encoded_image = tf.image.encode_jpeg(pred_image)
    tf.io.write_file(output_path, encoded_image)
    
    return output_path

# 为提交准备预测
def prepare_submission():
    import zipfile
    
    # 创建输出目录
    os.makedirs("generated_images", exist_ok=True)
    
    # 获取所有测试照片
    test_photos = glob(os.path.join(PHOTO_DIR, "*.jpg"))
    
    # 为每张照片生成莫奈风格版本
    for i, photo_path in enumerate(test_photos):
        filename = os.path.basename(photo_path)
        output_path = os.path.join("generated_images", filename)
        predict(photo_path, output_path)
        
        # 打印进度
        if (i+1) % 10 == 0:
            print(f"已处理 {i+1}/{len(test_photos)} 张图片")
    
    # 创建提交用的ZIP文件
    with zipfile.ZipFile("submission.zip", "w") as zip_file:
        for file in glob("generated_images/*.jpg"):
            zip_file.write(file, os.path.basename(file))
    
    print("已创建提交文件：submission.zip")

if __name__ == "__main__":
    # 创建检查点目录
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 训练模型
    train()
    
    # 生成提交文件
    prepare_submission() 