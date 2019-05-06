import dlib
import pickle
import cv2


class Detector:
    def __init__(self, shape_model_file, face_rec_model_file, gender_pickle_file):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_model_file)
        self.recognizer = dlib.face_recognition_model_v1(face_rec_model_file)
        self.classifier = pickle.load(open(gender_pickle_file, 'r'))

    def predict_gender(self, encoding, thresh=0.4):
        result = self.classifier(dlib.vector(encoding))        
        if result > thresh:
            return "male"
        if result < -thresh:
            return "female"
        return "unknown"

    def face_descriptor(self, img, rect):
        return self.recognizer.compute_face_descriptor(img, self.predictor(img, rect), 1)

    def face_size(self, rect):
        return rect.width() * rect.height()

    def process(self, image):
        image = cv2.resize(image, (256, 256))
        dets, scores, _ = self.detector.run(image, 1, -1)
        if sum(scores) < 0.1:
            return 'unknown'

        faces = list(dets)
        faces.sort(cmp=lambda x, y: self.face_size(y) - self.face_size(x))
        face = faces[0]
        description = self.face_descriptor(image, face)
        return self.predict_gender(description)
    
    def process_description(self, image):
        image = cv2.resize(image, (256, 256))
        dets, scores, _ = self.detector.run(image, 1, -1)
        if sum(scores) < 0.1:
            return

        faces = list(dets)
        faces.sort(cmp=lambda x, y: self.face_size(y) - self.face_size(x))
        face = faces[0]
        description = self.face_descriptor(image, face)
        return description
