import cv2
from definitions import ROOT_DIR
import numpy as np


from preprocessing import PreProcessor
# from classifier import FaceClassifier
# from embedded_data import Embedding

# ---
from cnn_classifier import FaceClassifier

from datetime import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

p = PreProcessor()
# embedder = Embedding()
# classifier = FaceClassifier()
# data = np.load('data/dataset.npz')
# label = np.load('data/label.npz')
#
# embeded = np.load('data/embedded_images.npz')
# label = label['arr_0']
# embeded = embeded['arr_0']
# --------------------------------------------
# classifier = FaceClassifier()
#
# classifier.set_model_path(ROOT_DIR + '/models/model.pkl')
# classifier.set_label_path(ROOT_DIR + '/models/labels.json')
# classifier.set_data_train_path(ROOT_DIR + '/data/embedded_faces.npz')
# classifier.train()
# label_dict = classifier.get_label_dict(ROOT_DIR + '/models/labels.json')
# -----------------------------------------------------------------
classifier = FaceClassifier()

# classifier.set_model_path(ROOT_DIR + '/models/model_09_02.h5')
classifier.set_model_path(ROOT_DIR + '/models/model_12_02.h5')
# classifier.set_label_path(ROOT_DIR + '/models/labels.json')
label_dict = classifier.get_label_dict(ROOT_DIR + '/models/labels.json')


face_detected = False
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Handles the mirroring of the current frame
    frame = cv2.flip(frame, 1)
    if p.extract_video_face(frame)  != None:
        face_detected = True
        faces = p.extract_video_face(frame)
        for (x1, x2, y1, y2) in faces:
            # print(x1, x2, y1, y2)
            # x1, x2, y1, y2 = p.extract_video_face(frame)
            cv2.rectangle(frame, (x1-8, y1-8), (x2+8, y2+8), (0, 0, 255), 2)
            pixels = np.asarray(frame)
            face = pixels[y1:y2, x1:x2]
            # resize pixels to the models size
            output_image = np.array(face)

            face_pixel = np.asarray(output_image)
            face_pixel = cv2.resize(face_pixel, (160, 160))

            predicted_label, accuracy = classifier.classify(face=face_pixel, label_dict=label_dict)
            if accuracy > 80:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = predicted_label
                color = (255, 255, 255)
                stroke = 1
                cv2.putText(frame, "%s (%.3f %%)" % (name, accuracy), (x1, y1 - 17), font, 1, color, thickness=stroke)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = "Unknown"
                color = (255, 255, 255)
                stroke = 1
                cv2.putText(frame, "%s (%.3f %%)" % (name, accuracy), (x1, y1-17), font, 1, color, thickness=stroke)

        # ------------------------


    else:
        face_detected = False

    k = cv2.waitKey(1)
    # SPACE pressed
    if k % 256 == 32 and face_detected:
        currentTime = datetime.now().strftime("%d%m%Y%H%M%S")
        name = './data/raw/hoang/frame_' + str(currentTime) + '.jpg'
        cv2.imwrite(name, frame)
        print('>> Saved image: ', name)

    # Display the resulting frame
    cv2.imshow('Streaming', frame)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    # To stop duplicate images

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
