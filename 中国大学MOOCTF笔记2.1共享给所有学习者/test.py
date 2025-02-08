import tensorflow as tf

tensorflow_version = tf.__version__
gpu_available = tf.test.is_gpu_available()
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
print("tensorflow version:", tensorflow_version, "\tGPU available:", gpu_available)

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([1.0, 2.0], name="b")
result = tf.add(a, b, name="add")
print(result)
