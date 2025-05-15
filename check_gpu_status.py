import tensorflow as tf

print("TensorFlow版本:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("可用的GPU设备:", gpus)
print("GPU是否可用:", len(gpus) > 0)

if len(gpus) > 0:
    print("\n当前训练正在使用GPU")
else:
    print("\n当前训练未使用GPU，正在使用CPU进行训练") 