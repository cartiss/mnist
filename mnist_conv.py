from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()  # (60000, 28, 28) (10000, 28, 28)

X_val = np.array(X_train[:10000])
X_train = np.array(X_train[10000:])
X_test = np.array(X_test)
y_test = np.array(y_test)
y_val = np.array(y_train[:10000])
y_train = np.array(y_train[10000:])

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(10)
])

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)

evaluation = model.evaluate(X_test, y_test)

model.save('models/mnist_conv.model')
