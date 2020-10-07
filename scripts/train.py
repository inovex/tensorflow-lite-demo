"""
Created by Robin Baumann <mail@robin-baumann.com> at 22.05.20.
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from model import mobilenet_qa


def main():
  (train_ds, val_ds, test_ds), metadata = tfds.load(
      'cars196',
      split=['train[:80%]', 'train[80%:]', 'test'],
      with_info=True,
      as_supervised=True
  )

  normalize = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

  def prepare(ds, shuffle=False, batch_size=32, num_classes=196):
    # Normalize images to [0, 1] and convert label to one-hot vector
    ds = ds.map(lambda sample: (
        tf.image.resize(
            normalize(sample[0]), (320, 320)),
        tf.one_hot(sample[1], num_classes)),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
      ds = ds.shuffle(1000)

    # Batch all datasets
    ds = ds.batch(batch_size)

    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  train_ds = prepare(train_ds, shuffle=True, batch_size=1)
  val_ds = prepare(val_ds, batch_size=1)
  # test_ds = prepare(test_ds)

  callbacks = [
      tf.keras.callbacks.ReduceLROnPlateau(verbose=1),
      tf.keras.callbacks.TensorBoard(log_dir="./logs")
  ]

  model = mobilenet_qa.get_model(in_shape=(320, 320, 3), num_classes=196,
                                 qat=True)
  model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
  model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)


if __name__ == "__main__":
  main()
