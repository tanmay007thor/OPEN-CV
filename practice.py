import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
model_cifar10 = models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model_cifar10.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)
split = 50000
x_val, y_val = x_train[:split], y_train[:split]
x_train, y_train = x_train[split:], y_train[split:]
model_mnist = models.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model_mnist.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cifar10 = model_cifar10.fit(train_images, train_labels, epochs=30, batch_size=128, validation_data=(test_images, test_labels))
history_mnist = model_mnist.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=(x_val, y_val))
test_loss_cifar10, test_acc_cifar10 = model_cifar10.evaluate(test_images, test_labels)
test_loss_mnist, test_acc_mnist = model_mnist.evaluate(x_test, y_test)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_cifar10.history['accuracy'], label='CIFAR-10 Training Accuracy')
plt.plot(history_cifar10.history['val_accuracy'], label='CIFAR-10 Validation Accuracy')
plt.title('CIFAR-10 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_mnist.history['accuracy'], label='MNIST Training Accuracy')
plt.plot(history_mnist.history['val_accuracy'], label='MNIST Validation Accuracy')
plt.title('MNIST Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()
print('The code is running.')
