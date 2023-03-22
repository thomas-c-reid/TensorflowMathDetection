from PIL import Image
import numpy as np
from numpy import asarray
import os
from sklearn.model_selection import train_test_split
import pickle

# TODO Remove X Y Z categoty, it mixes up the

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "add", "dec", "eq", "div", "mul", "sub", "none"]
DATADIR = "C:/Users/realt/Documents/Datasets/HandwrittenMathSymbols/dataset/"
tempImgLink = "C:/Users/realt/Documents/Datasets/HandwrittenMathSymbols/dataset/0/0CdBlhLw.png"
img_size = (64, 64)

train_data = []
train_labels = []
test_data = []
test_labels = []
val_data = []
val_labels = []


def preprocessData():
    # selects each image from each file within each digits directory
    # it converts the image to greyscale, resizes it,
    for category in CATEGORIES:
        resized_images = []
        path = f"{DATADIR}{category}"
        for img in os.listdir(path):
            path2 = f"{path}/{img}"
            if path2[-3:] != "ory":
                temp2 = Image.open(path2).convert("L")
                resized_image = temp2.resize(img_size, resample=Image.BILINEAR)
                numpy_image = asarray(resized_image)
                resized_images.append(numpy_image)


        X_train, X_rem = train_test_split(resized_images, train_size=0.7)
        X_test, X_validation = train_test_split(X_rem, train_size=0.5)

        for img in X_train:
            train_data.append(img)
        for img in X_test:
            test_data.append(img)
        for img in X_validation:
            val_data.append(img)
        for label in range(len(X_train)):
            train_labels.append(category)
        for label in range(len(X_test)):
            test_labels.append(category)
        for label in range(len(X_validation)):
            val_labels.append(category)


preprocessData()

pickleNameList = ["train_data.pickle", "train_labels.pickle", "test_data.pickle", "test_labels.pickle",
                  "val_data.pickle",
                  "val_labels.pickle"]
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)

pickle_out = open(pickleNameList[0], "wb")
pickle.dump(train_data, pickle_out)
pickle_out.close()

pickle_out = open(pickleNameList[1], "wb")
pickle.dump(train_labels, pickle_out)
pickle_out.close()

pickle_out = open(pickleNameList[2], "wb")
pickle.dump(test_data, pickle_out)
pickle_out.close()

pickle_out = open(pickleNameList[3], "wb")
pickle.dump(test_labels, pickle_out)
pickle_out.close()

pickle_out = open(pickleNameList[4], "wb")
pickle.dump(val_data, pickle_out)
pickle_out.close()

pickle_out = open(pickleNameList[5], "wb")
pickle.dump(val_labels, pickle_out)
pickle_out.close()