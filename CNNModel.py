from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from google.colab import files

# iMPORTING DATA ===========================================================================
def readPickleData():
    file_path = "./processed data"
    data_dict = {}
    if os.path.exists(file_path):
        pickle_dir = os.path.join(os.path.dirname(__file__), file_path[2:])
        pickle_files = [f for f in os.listdir(pickle_dir) if f.endswith('.pickle')]

        for i, pickle_file in enumerate(pickle_files):
            with open(os.path.join(pickle_dir, pickle_file), 'rb') as f:
                data = pickle.load(f)
                data_dict[f'data_{i + 1}'] = data
    else:
        with open('test_data.pickle', 'rb') as p:
            test_data = pickle.load(p)
        with open('test_labels.pickle', 'rb') as i:
            test_labels = pickle.load(i)
        with open('train_data.pickle', 'rb') as c:
            train_data = pickle.load(c)
        with open('train_labels.pickle', 'rb') as k:
            train_labels = pickle.load(k)
        with open('val_data.pickle', 'rb') as l:
            val_data = pickle.load(l)
        with open('val_labels.pickle', 'rb') as e:
            val_labels = pickle.load(e)

            test_data = np.reshape(test_data, (test_data.shape[0], 64, 64, 1))
            train_data = np.reshape(train_data, (train_data.shape[0], 64, 64, 1))
            val_data = np.reshape(val_data, (val_data.shape[0], 64, 64, 1))

            data_dict['data_1'] = test_data
            data_dict['data_2'] = test_labels
            data_dict['data_3'] = train_data
            data_dict['data_4'] = train_labels
            data_dict['data_5'] = val_data
            data_dict['data_6'] = val_labels

        return data_dict


data_dict = readPickleData()

test_data = data_dict["data_1"]
test_data = tf.keras.utils.normalize(test_data, axis=1)
test_labels = data_dict["data_2"]
train_data = data_dict["data_3"]
train_data = tf.keras.utils.normalize(train_data, axis=1)
train_labels = data_dict["data_4"]
val_data = data_dict["data_5"]
val_data = tf.keras.utils.normalize(val_data, axis=1)
val_labels = data_dict["data_6"]


# CONVERTING STRING TYPE LABELS TO INT
def convert_strings_to_integers(array):
    # Create a dictionary to map strings to index integers
    string_to_index = {
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
        "add": 10,
        "dec": 11,
        "eq": 12,
        "div": 13,
        "mul": 14,
        "sub": 15,
        "x": 16,
        "y": 17,
        "z": 18,
    }

    # Loop through the array and convert each string to its index integer
    integer_array = []
    for string in array:
        integer_array.append(string_to_index[string])

    return integer_array


train_labels = convert_strings_to_integers(train_labels)
test_labels = convert_strings_to_integers(test_labels)
val_labels = convert_strings_to_integers(val_labels)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
val_labels = to_categorical(val_labels)


# BUILDING THE MODEL ===========================================================================
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64,1)),
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
                    epochs=40,
                    validation_data=(val_data, val_labels))

model.save('DigitClassifier.h5')