import numpy as np
from keras.models import load_model
from definitions import ROOT_DIR
from tqdm import tqdm
from datetime import datetime


class Embedding:

    def __init__(self):
        self.model = load_model(ROOT_DIR + '/keras-facenet/model/facenet_keras.h5')

    # get the face embedding for one face
    def get_embedding(self, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std

        face_pixels = np.expand_dims(face_pixels, axis=0)

        # make prediction to get embedding
        yhat = self.model.predict(face_pixels)
        return yhat[0]


if __name__ == '__main__':
    e = Embedding()

    data = np.load(ROOT_DIR + '/data/faces_084753_07Feb2020.npz', allow_pickle=True)
    trainX = data['arr_0']
    trainY = data['arr_1']
    print('Loading extracted face')
    print(trainX.shape)
    print(trainY.shape)
    print('-----------------------------------')
    print('Embedding face using facenet and save')
    # convert each face in the train set to an embedding
    newTrainX = list()
    for face in tqdm(trainX, desc='Embedding train set'):
        embedding = e.get_embedding(face)
        newTrainX.append(embedding)
    newTrainX = np.array(newTrainX)

    # currentTime = datetime.now().strftime("%H%M%S_%d%b%Y")
    # np.savez_compressed(ROOT_DIR + '/data/embedded_faces_{}.npz'.format(currentTime), newTrainX, trainY)
    np.savez_compressed(ROOT_DIR + '/data/embedded_faces.npz', newTrainX, trainY)
    print("Done ... :)")
