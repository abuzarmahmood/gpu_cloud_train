"""
Train network to classify images from wikiart dataset
1) Load images and split into train+test sets
2) Download InceptionV3 model and add new dense layers at end
3) Train on CPU/GPU and print out training metrics
"""

#import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
#data_dir = pathlib.Path(data_dir)
data_dir = '/home/abuzarmahmood/Desktop/img_conv_net/download_data/flower_photos'

used_device = tf.config.list_physical_devices()
print(used_device)

# Check if in docker container
file_path = '/proc/1/cgroup'
lines = open(file_path,'r').readlines()
if 'docker' in "".join(lines):
    mirrored_strategy = tf.distribute.MirroredStrategy()

batch_size = 32
img_height = 64
img_width = 64

from_dir_kwargs = dict(
		directory = data_dir,
		  validation_split=0.2,
		  seed=123,
		  image_size=(img_height, img_width),
		  batch_size=batch_size)

train_ds = \
	tf.keras.utils.image_dataset_from_directory(
	  subset="training", **from_dir_kwargs)

val_ds = \
    tf.keras.utils.image_dataset_from_directory(
      subset="validation", **from_dir_kwargs)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#AUTOTUNE = tf.data.AUTOTUNE
#
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#normalization_layer = layers.Rescaling(1./255)
#
#normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#image_batch, labels_batch = next(iter(normalized_ds))
#first_image = image_batch[0]
## Notice the pixel values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
    layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs_range = range(epochs)
#
#plt.figure(figsize=(8, 8))
#plt.subplot(1, 2, 1)
#plt.plot(epochs_range, acc, label='Training Accuracy')
#plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#plt.legend(loc='lower right')
#plt.title('Training and Validation Accuracy')
#
#plt.subplot(1, 2, 2)
#plt.plot(epochs_range, loss, label='Training Loss')
#plt.plot(epochs_range, val_loss, label='Validation Loss')
#plt.legend(loc='upper right')
#plt.title('Training and Validation Loss')
#plt.show()
