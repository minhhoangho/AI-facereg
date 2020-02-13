import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from definitions import TRAIN_DIR, TEST_DIR
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator


class PreProcessor:
    # load images and extract faces for all images in a directory

    def __init__(self):
        # create the detector, using default weights
        self.detector = MTCNN()

    def extract_face(self, filename, required_size=(160, 160)):
        try:
            # load image from file
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # convert to array
            pixels = np.asarray(image)
            # detect faces in the image
            results = self.detector.detect_faces(pixels)
            # extract the bounding box from the first face
            x1, y1, width, height = results[0]['box']
            # x1, y1, width, height = x1 + 2, y1 + 2, width - 2, height - 2
            # bug fix
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            # extract the face
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the models size
            output_image = np.array(face)

            face_array = np.asarray(output_image)
            face_array = cv2.resize(face_array, required_size)
            return face_array
        except:
            return None

    def extract_video_face(self, frame):
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # convert to array
            pixels = np.asarray(image)
            faces = list()
            # detect faces in the image
            results = self.detector.detect_faces(pixels)
            for result in results:
                # extract the bounding box from each face
                x1, y1, width, height = result['box']
                # bug fix
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                faces.append((x1, x2, y1, y2))
            return faces
        except:
            return None

    def load_faces(self, directory):
        faces = []
        i = 1
        # enumerate files
        for filename in tqdm(os.listdir(directory), desc='Loading images'):
            # path
            path = directory + filename
            # get face
            face = self.extract_face(path)
            print(i, face.shape)
            if face is None:
                continue
            faces.append(face)
            # plot
            plt.subplot(3, 7, i)
            plt.axis('off')
            plt.imshow(face)

            i += 1
        plt.show()
        return np.array(faces)

    def load_dataset(self, directory):
        X = []
        Y = []
        j = 1
        for subdir in os.listdir(directory):

            # path
            path = directory + '/' + subdir + '/'
            if not os.path.isdir(path):
                continue
            # load all faces in sub directory
            faces = self.load_faces(path)
            # create labels
            labels = [subdir for _ in range(len(faces))]
            # store
            X.extend(faces)
            Y.extend(labels)
            print('\n>>Loaded %d examples for class: %s' % (len(faces), subdir))
            print('--------------------------------------------------------------------\n')
        return np.asarray(X), np.asarray(Y)


if __name__ == '__main__':
    # currentTime = datetime.now().strftime("%d%m%Y_%H%M%S")
    # print(currentTime)
    p = PreProcessor()
    print('Extracting faces and saving ...')

    X_train, Y_train = p.load_dataset(TRAIN_DIR)
    # x_test, y_test = p.load_dataset(TEST_DIR)
    # Save dataset and labels
    # currentTime = datetime.now().strftime("%H%M%S_%d%b%Y")
    # np.savez_compressed('data/faces_{}.npz'.format(currentTime), X_train, Y_train)
    np.savez_compressed('data/faces.npz', X_train, Y_train)

    print("Done ... :)")
