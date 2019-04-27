import pickle
import cv2
import glob
import numpy as np
from keras.models import load_model
from train import image_to_feature_vector

if __name__ == "__main__":
    model = load_model('model.nn')
    classes = pickle.load(open('classes.pickle', 'r'))

    for image_file in glob.glob('samples/**/*.jpg'):
        image = cv2.imread(image_file)
        image = cv2.resize(image, (256, 256))
        features = image_to_feature_vector(image) / 255.0
        features = np.array([features])
        probs = model.predict(features)[0]
        prediction = probs.argmax(axis=0)
        label = "{}: {:.2f}%".format(
            classes[prediction],
            probs[prediction] * 100,
        )
        cv2.putText(
            image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2,
        )
        cv2.imshow("Image", image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
