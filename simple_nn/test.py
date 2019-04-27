import pickle
import cv2
import os
import glob
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from architecture import image_size

if __name__ == "__main__":
    model = load_model(os.path.join('bin', 'model.nn'))
    classes = pickle.load(open(os.path.join('bin', 'classes.pickle'), 'r'))

    for image_file in glob.glob('samples/**/*.jpg'):
        image = cv2.imread(image_file)
        original = image.copy()
        original = cv2.resize(original, (512, 512))
        image = cv2.resize(image, image_size)
        features = img_to_array(image) / 255.0
        features = np.array([features])
        probs = model.predict(features)[0]
        prediction = probs.argmax(axis=0)
        label = "{}: {:.2f}%".format(
            classes[prediction],
            probs[prediction] * 100,
        )        
        cv2.putText(
            original, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, (0, 255, 0), 2,
        )
        cv2.imshow("image", original)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
