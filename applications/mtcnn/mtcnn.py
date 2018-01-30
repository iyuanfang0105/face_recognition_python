# coding=utf-8
import tensorflow as tf
import detect_face
import cv2
import numpy as np

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709


def mtcnn_face_dection_from_image(img_path):
    image = cv2.imread(img_path)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    with tf.Session() as sess:
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        bounding_boxes, landmarks = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    faces_num = bounding_boxes.shape[0]  # number of faces
    print "====>>>> Found faces：" + str(faces_num)

    for face_position, landmark_position in zip(bounding_boxes, np.transpose(landmarks)):
        face_position = face_position.astype(int)
        landmark_position = landmark_position.astype(int)
        cv2.rectangle(image, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
        for x, y in zip(landmark_position[0::2], landmark_position[1::2]):
            cv2.circle(image, (y, x), 2, (255, 0, 0), 2)
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)

if __name__ == '__main__':
    image = "../../images/faces.jpg"
    mtcnn_face_dection_from_image(image)