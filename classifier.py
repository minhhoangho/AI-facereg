from definitions import ROOT_DIR
import numpy as np
import random
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn import preprocessing as p
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from embedded_data import Embedding
from datetime import datetime

import json
import pickle
import logging

logger = logging.getLogger(__name__)


class FaceClassifier:
    # models linear support vector classifier
    def __init__(self, data_train_path=None, embedded_faces_path=None, model_path=None):

        # self.Normalizer = Normalizer(norm='l2')
        # self.LabelEncoder = LabelEncoder()
        self.model = SVC(kernel='linear', probability=True)

        self.embedder = Embedding()
        self.embedded_faces_path = embedded_faces_path
        self.input_shape = (160, 160, 3)
        self.model_path = model_path
        self.label_path=None
        self.num_class = None
        self.inputX = None
        self.outputY = None
        self.data_train_path = data_train_path

    def set_model_path(self, path):
        self.model_path = path

    def set_data_train_path(self, path):
        self.data_train_path = path
        self.inputX, self.outputY = self.load_data()

    def set_label_path(self, path):
        self.label_path = path

    def get_label_dict(self, path):
        if self.label_path:
            self.label_path = path
        with open(path, 'r', encoding='utf-8') as file:
            label_dict = json.loads(file.read())
        label_dict = dict(label_dict)
        label_dict = {int(key): str(val) for key, val in label_dict.items()}
        self.num_class = len(label_dict)
        return label_dict

    def load_data(self):
        """
           Load data
           :return: 4D-array X and 2D-array y
        """
        data = np.load(self.data_train_path)

        X, y = data['arr_0'], data['arr_1']

        # normalize X
        normalizer = Normalizer(norm='l2')
        X = normalizer.transform(X)

        # labeling
        encoder = LabelEncoder()

        labels = y
        y = encoder.fit_transform(labels)

        label_dict = dict(zip(y, labels))
        label_dict = {str(key): str(val) for key, val in label_dict.items()}
        with open(self.label_path, 'w') as outfile:  # dumps labels to json file
            json.dump(label_dict, outfile)

        self.num_class = len(label_dict)

        return np.array(X), np.array(y)

    def load_model(self):
        """
        Load model from file
        :return: None
        """
        # self.model = pickle.load(open(self.model_path, 'rb'))

        # Load from file
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)


    def train(self):
        self.model.fit(self.inputX, self.outputY)

         # Save to file
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)

        # pickle.dump(self.model, open(self.model_path, 'wb'))
        # self.model.save_weights('./models/MobileNetV2_{}.h5'.format(current_time))

    def predict(self, X):

        if self.model is None:
            self.load_model()

        X = np.array(X)
        return self.model.predict(X)

    def classify(self, face, label_dict=None):
        """
            Classify images
            :param face: input of one face
            :param label_dict: dictionary of label ids and names
            :return: label array
        """
        X = self.embedder.get_embedding(face_pixels=face)
        y = self.predict([X])
        id = y[0]
        prob = self.model.predict_proba([X])
        prob = prob[0, id] * 100

        return label_dict[id], prob

if __name__ == '__main__':
    classifier = FaceClassifier()

    classifier.set_model_path(ROOT_DIR + '/models/model.pkl')
    classifier.set_label_path(ROOT_DIR + '/models/labels.json')
    classifier.set_data_train_path(ROOT_DIR + '/data/embedded_faces_094643_07Feb2020.npz')
    #
    classifier.train()
    X = classifier.inputX[3]
    y = classifier.model.predict([X])
    prob = classifier.model.predict_proba([X])
    prob = prob[0, y[0]] * 100
    print(y, prob)
