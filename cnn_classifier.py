from definitions import ROOT_DIR
import numpy as np
from sklearn.preprocessing import Normalizer, LabelEncoder
from matplotlib import pyplot as plt
from embedded_data import Embedding
import json
import cv2
import keras
from sklearn.model_selection import train_test_split

class FaceClassifier:
    # models linear support vector classifier
    def __init__(self, data_train_path=None, embedded_faces_path=None, model_path=None):

        # self.Normalizer = Normalizer(norm='l2')
        # self.LabelEncoder = LabelEncoder()
        self.model = None
        self.label_path=None
        self.embedder = Embedding()
        self.embedded_faces_path = embedded_faces_path
        self.input_shape = (160, 160, 3)
        self.model_path = model_path
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


        # for idx, image in enumerate(X):
        #     X[idx] =image[:, :, [2, 1, 0]]

        # normalize X
        # normalizer = Normalizer(norm='l2')
        # X = normalizer.transform(X)

        # labeling
        encoder = LabelEncoder()

        labels = y
        y = encoder.fit_transform(labels)

        label_dict = dict(zip(y, labels))
        label_dict = {str(key): str(val) for key, val in label_dict.items()}
        with open(self.label_path, 'w') as outfile:  # dumps labels to json file
            json.dump(label_dict, outfile)

        self.num_class = len(label_dict)

        y_train = np.zeros((len(y), self.num_class), dtype="float")
        for idx, val in enumerate(y):
            y_train[idx][val] = 1
        y = y_train
        return np.array(X), np.array(y)

    def load_model(self):
        """
        Load model from file
        :return: None
        """
        self.model = self.build_model()
        self.model.load_weights(self.model_path)

    def build_model(self):
        # khai bao input layer
        input_layer = keras.layers.Input(shape=self.input_shape, name='input')

        # su dung pre-trained model
        pretrained_model = keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet')
        pretrained_model_output = pretrained_model(input_layer)
        global_avg = keras.layers.GlobalAveragePooling2D()(pretrained_model_output)

        #   # su dung pre-trained model
        # pretrained_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
        # pretrained_model_output = pretrained_model(input_layer)
        # global_avg = keras.layers.GlobalAveragePooling2D()(pretrained_model_output)

        # fully-connected layer 1
        dense = keras.layers.Dense(units=512)(global_avg)
        dense = keras.layers.BatchNormalization()(dense)
        dense = keras.layers.ReLU()(dense)
        output = keras.layers.Dense(units=self.num_class)(dense)
        output = keras.layers.Activation('softmax')(output)
        model = keras.models.Model(input_layer, output)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(),
                      metrics=['accuracy'])
        print(model.summary())

        return model

    def train(self):
        self.model = self.build_model()
        x_train, x_test, y_train, y_test = train_test_split(self.inputX, self.outputY, test_size=0.33)
        self.model.fit(x_train, y_train, batch_size=5, epochs=50)
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        self.model.save_weights(self.model_path)

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
        # X = self.embedder.get_embedding(face_pixels=face)
        y = self.predict([face])
        result = np.argmax(y, axis=1)[0]

        prob = y[0][result] * 100

        return label_dict[result], prob


if __name__ == '__main__':
    classifier = FaceClassifier()

    classifier.set_model_path(ROOT_DIR + '/models/model_12_02.h5')
    classifier.set_label_path(ROOT_DIR + '/models/labels.json')
    classifier.set_data_train_path(ROOT_DIR + '/data/faces.npz')
    # x, y = classifier.load_data()
    #
    classifier.train()
    print("Done")

    # label_dict = classifier.get_label_dict(ROOT_DIR + '/models/labels.json')
    # test1X = np.load(classifier.data_train_path)
    # test1X = test1X['arr_0']
    # img_test = test1X[3]
    # cv2.imshow('img',img_test)
    #
    # X = classifier.inputX[3]
    # label, prob = classifier.classify(X, label_dict)
    # print(label)
    # print(prob)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
