import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# TODO more pre processing to reduce image size, causing problems with NN


def readPickleData(name):
    pickle_dir = os.path.join(os.path.dirname(__file__), name)
    pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pickle')]
    data_dict = {}

    for i, pickle_file in enumerate(pickle_files):
        with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
            data = pickle.load(f)
            data_dict[f'data_{i + 1}'] = data

    return data_dict


data_dict = readPickleData('processed data')

test_data = data_dict["data_1"]
test_data = tf.keras.utils.normalize(test_data, axis=1)
test_labels = data_dict["data_2"]
train_data = data_dict["data_3"]
train_data = tf.keras.utils.normalize(train_data, axis=1)
train_labels = data_dict["data_4"]
val_data = data_dict["data_5"]
val_data = tf.keras.utils.normalize(val_data, axis=1)
val_labels = data_dict["data_6"]


# BUILDING THE MODEL

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Conv2D(256, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(19, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train_data, train_labels,
                    batch_size=32,
                    epochs=10,
                    validation_data=(val_data, val_labels))

model.save('DigitClassifier.h5')