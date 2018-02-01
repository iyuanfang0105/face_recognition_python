# coding=utf-8
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

import MTCNN_face_detector


def face_detect_from_image(mtcnn_detector, image):
    bounding_boxes, landmarks = mtcnn_detector.detect_face(image)
    faces_num = bounding_boxes.shape[0]  # number of faces
    print "====>>>> Found faces：" + str(faces_num)
    faces = []
    points = []
    for face_position, landmark_position in zip(bounding_boxes, np.transpose(landmarks)):
        face_position = face_position.astype(int)[0:4]
        landmark_position = landmark_position.astype(int)
        faces.append(face_position)
        # aligned_img, pos = aligner.align(160, image, landmark_position)
        temp = []
        for x, y in zip(landmark_position[0:5], landmark_position[5:]):
            temp.append((x, y))
        points.append(temp)
    return faces, points


def face_detect_from_camera(mtcnn_detector):
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        count = count + 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        if count % 2 == 0:
            faces, points = face_detect_from_image(mtcnn_detector, frame)
            plot_faces_landmarks(frame, faces, points, plt_flag=False)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def plot_faces_landmarks(image, faces, points, plt_flag=True):
    '''
    plot the detected faces and landmarks
    :param image: 3D array
    :param faces:
    :param points:
    :return:
    '''
    for face, point in zip(faces, points):
        cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)
        for p in point:
            cv2.circle(image, (p[0], p[1]), 4, (0, 0, 255), 2)
    if plt_flag:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == '__main__':
    image_path = "../../images/faces.jpg"
    image = cv2.imread(image_path)
    mtcnn_detector = MTCNN_face_detector.MTCNN()
    # faces, points = face_detect_from_image(mtcnn_detector, image)
    # plot_faces_landmarks(image, faces, points)
    face_detect_from_camera(mtcnn_detector)