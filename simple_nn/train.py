import cv2
import os
import glob
import random
import pickle
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from architecture import LeNet, SmallerVGGNet, image_size


def read_files(image_files_pattern, limit=100, size=(28, 28)):
    image_files = list(glob.glob(image_files_pattern))
    random.shuffle(image_files)

    if len(image_files) > limit:
        image_files = image_files[:limit]

    data = []
    labels = []
    for image_file in image_files:
        label = image_file.split(os.path.sep)[1]

        image = cv2.imread(image_file)
        image = cv2.resize(image, size)
        image = img_to_array(image)

        data.append(image)
        labels.append(label)

    le = LabelEncoder()
    labels = le.fit_transform(labels)    
    data = np.array(data, dtype="float") / 255.0    
    labels = np_utils.to_categorical(labels, 3)
    return data, labels, le.classes_


if __name__ == "__main__":
    data, labels, classes = read_files(
        'labels/**/*.jpg',
        limit=2000,
        size=image_size,
    )
    model = LeNet(width=image_size[1], height=image_size[0], depth=3, classes=3)
    loss, accuracy = model.fit(data, labels)
    print "loss={:.2f}, accuracy: {:.2f}%".format(loss, accuracy * 100)

    if not os.path.exists('bin'):
        os.mkdir('bin')

    model.save(os.path.join('bin', 'model.nn'))
    pickle.dump(classes, open(os.path.join('bin', 'classes.pickle'), 'w'))
