import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

keras_model_path = './save'
another_strategy = tf.distribute.OneDeviceStrategy('/cpu:0')

def get_data():
  datasets = tfds.load(name='mnist', as_supervised=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  BUFFER_SIZE = 10000

  BATCH_SIZE_PER_REPLICA = 64
  BATCH_SIZE = BATCH_SIZE_PER_REPLICA * another_strategy.num_replicas_in_sync

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

  train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

  return train_dataset, eval_dataset


train_dataset, eval_dataset = get_data()

restored_keras_model_ds = tf.keras.models.load_model(keras_model_path)
history = restored_keras_model_ds.fit(train_dataset, epochs=2)
restored_keras_model_ds.summary()
acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='test Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Validation Loss')
plt.title('Validation Loss')
plt.legend()
plt.show()

