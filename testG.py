import tensorflow as tf

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU {i} обнаружен: {gpu}")
else:
    print("❌ GPU не обнаружено, используется CPU")