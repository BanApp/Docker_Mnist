import os
import tensorflow as tf
import random
import numpy as np
from sklearn.model_selection import train_test_split


def normalize_images(images):
    mean = np.mean(images)
    std = np.std(images)
    images = (images - mean) / std
    return images


images = []
labels = []

for i in range(10):
    folder_name = str(i)
    folder_path = os.path.join("100_1/" + folder_name)
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        image = tf.keras.preprocessing.image.load_img(image_path)
        image = np.array(image)
        images.append(image)
        labels.append(folder_name)

images = np.array(images)
labels = np.array(labels)

images = normalize_images(images)

(train_images, test_images, train_labels, test_labels) = train_test_split(images, labels, test_size=0.2)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

loss, accuracy = model.evaluate(test_images, test_labels)

print('loss:', loss)
print('accuracy:', accuracy)
