from read_images import read_idx
from cnn import build_cnn_model

import time
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU and force CPU use


class TimeHistory(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1} took {time.time() - self.epoch_time_start} seconds")

# Paths to your files
images_path = './data/train-images-idx3-ubyte/train-images-idx3-ubyte'
labels_path = './data/train-labels-idx1-ubyte/train-labels-idx1-ubyte'

images = read_idx(images_path)
labels = read_idx(labels_path)

print(images.shape)  # Should show (number of images, 28, 28)
print(labels.shape)  # Should show (number of labels,)

# build the model
cnn_model = build_cnn_model()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preprocess the images for TensorFlow input (normalize and reshape)
images = images.reshape((-1, 28, 28, 1))  # Add the channel dimension - the 1 signals that these are single-channel grayscale images
images = images / 255.0  # Normalize pixel values to 0-1

# Train the model using your preprocessed data
cnn_model.fit(images, labels, epochs=5, batch_size=64, validation_split=0.2)
cnn_model.save('mnist_cnn_model.h5')