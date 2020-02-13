from keras.models import load_model
import numpy as np
import random
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn import preprocessing as p
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from embedded_data import Embedding
class FaceClassifier:

    def __init__(self, dataset_path=None, embedded_faces_path=None):
        # models linear support vector classifier
        self.model = SVC(kernel='linear', probability=True)
        self.Normalizer = Normalizer(norm='l2')
        self.LabelEncoder = LabelEncoder()
        self.dataset_path = dataset_path
        self.embedded_faces_path = embedded_faces_path
        self.embedder = Embedding()

    def load(self, embedding_path):
        return np.load(embedding_path);

        # data = np.load(embedding_path)
        # self.trainX, self.trainY, self.testX, self.testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        # print('Dataset: train=%d, test=%d' %( self.trainX.shape[0], self.testX.shape[0]))

    def normalize(self, input):
        return self.Normalizer.transform(input)

    def label_encoding(self, label):
        self.LabelEncoder.fit(label)
        return self.LabelEncoder.transform(label)

    def c_predict(self, normalized_input):
        return self.model.predict(normalized_input)

    def c_train(self, normalized_input, encoded_label):
        self.model.fit(normalized_input, encoded_label)
        print('>> Finished')

    def train(self, input, label):
        """
        train models with input and label

        :param input: embedded images as numpy array (num_image, width, height, color_channel)
        :param label: numpy array of string
        :return: None
        """

        # normalize input
        input = self.normalize(input)
        # label encoding expected output
        label = self.label_encoding(label)
        # train models
        self.c_train(input, label)

    def predict(self, input):
        """
        predict input base on models trained
        :param input: embedded images as numpy array (num_image, width, height, color_channel)
        :return: array of label predicted (encode format)
        """

        # normalize input
        input = self.normalize(input)
        return self.c_predict(input)

    def predict_one(self, face_pixels):
        """
        :param face_pixels: one image (width, height, color_channel)
        :return: label predicted
        """
        # print(face_pixels)
        embedded_img = self.embedder.get_embedding(face_pixels)
        label = np.load('data/label.npz')
        label = label['arr_0']

        out_encoder = LabelEncoder()

        out_encoder.fit(label)
        label = out_encoder.transform(label)


        # expand dimension of random_face_emb because we choose an image from image list
        samples = np.expand_dims(embedded_img, axis=0)

        predicted_class = self.predict(samples)
        prob = self.model.predict_proba(samples)

        class_index = predicted_class[0]
        #print('>> Class: ', class_index)
        class_probability = prob[0, class_index] * 100
        #print('>> Accuracy: ', class_probability)

        predict_name = out_encoder.inverse_transform(predicted_class)

        #print('Predicted: %s (%.3f)' % (predict_name[0], class_probability))
        return predict_name[0], class_probability
        # plot image
        # plt.imshow(face_pixels)
        # title = '%s (%.3f)' % (predict_name[0], class_probability)
        # plt.title(title)
        # plt.show()

    def test(self, input, expected_output):
        """
        :param input:  embedded images as numpy array (num_image, width, height, color_channel)
        :param expected_output: numpy array of string
        :return: None
        """
        # # normalize input
        # input = self.normalize(input)
        # # label encoding expected output
        # expected_output = self.label_encoding(expected_output)
        # predict result
        result = self.predict(input)
        accuracy = accuracy_score(self.label_encoding(expected_output), result)
        print('>> Accuracy:  %.3f' % (accuracy * 100))
        print('-----------------------------------')

    def run(self):


        # data = np.load(self.embedded_faces_path, allow_pickle=True)
        # print(data)
        #
        # trainX, trainY, testX, testY = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        # self.train(trainX, trainY)
        # self.test(testX, testY)
        # -----------------------------------------------------------------


        # Test


        data = np.load('data/dataset.npz')
        label = np.load('data/label.npz')

        embeded = np.load('data/embedded_image.npz')
        label = label['arr_0']
        embeded = embeded['arr_0']
        self.train(embeded, label)

        testX_faces = data['arr_0']

        # test models on a random example from the test dataset


        selection = random.choices([i for i in range(len(testX_faces))])[0]

        random_face_pixels = testX_faces[selection]

        self.predict_one(random_face_pixels)

        exit(0)




if __name__ == '__main__':
    cl = FaceClassifier('data/5-celebrity-faces-dataset.npz', 'data/5-celebrity-faces-embeddings.npz')
    cl.run()
