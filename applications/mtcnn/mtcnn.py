# coding=utf-8
import tensorflow as tf
import detect_face
import cv2
import numpy as np
from matplotlib import pyplot as plt

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709


def mtcnn_face_detection_from_image(image):
    '''
    face detection using mtcnn
    :param image: 3-D ararry
    :return:
    '''
    # image = cv2.imread(img_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    with tf.Session() as sess:
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        bounding_boxes, landmarks = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    faces_num = bounding_boxes.shape[0]  # number of faces
    # print "====>>>> Found faces：" + str(faces_num)
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


def mtcnn_face_detection_from_camera():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, points = mtcnn_face_detection_from_image(frame)
        # Display the resulting frame
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
            cv2.circle(image, (p[0], p[1]), 2, (255, 0, 0), 2)
    if plt_flag:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == '__main__':
    image_path = "../../images/faces.jpg"
    image = cv2.imread(image_path)
    faces, points = mtcnn_face_detection_from_image(image)
    plot_faces_landmarks(image, faces, points)
    print flag