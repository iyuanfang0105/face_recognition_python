import cv2
import tools


def get_detectors(face_config, eye_config=None):
    detectors = {}
    if face_config is not None:
        detectors["face"] = cv2.CascadeClassifier(face_config)
    if eye_config is not None:
        detectors["eye"] = cv2.CascadeClassifier(eye_config)
    return detectors


def detect_faces(image, face_detector=None, eye_detector=None):
    """
    detect faces form an image
    :param image: rgb image
    :param face_detector:
    :param eye_detector:
    :return:
    """
    if face_detector is None:
        print("Error: no face detector input, Please give the detector of face or eye")
        return -1
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(img_gray, 1.1, 3)
    # faces_roi = []
    for (x, y, w, h) in faces:
        # img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = img_gray[y:y + h, x:x + w]
        # roi_color = image[y:y + h, x:x + w]
        if eye_detector is not None:
            eyes = eye_detector.detectMultiScale(roi_gray)
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return faces


def mark_faces(image, faces_locate):
    for (x, y, w, h) in faces_locate:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    tools.show_images([image])


def cut_faces(image, faces_locate):
    faces = []
    for (x, y, w, h) in faces_locate:
        faces.append(image[y:y+h, x:x+w])
    return faces


if __name__ == '__main__':
    # face detector config: Viola-Jones face detector
    XMLTarget = '/home/meizu/WORK/software/opencv3.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'
    detectors = get_detectors(XMLTarget)
    # test image
    test_image_path = '../images/faces.jpg'
    img = cv2.imread(test_image_path)
    # tools.show_images([img, img_gray], titles=['original', 'gray'])

    faces_locate = detect_faces(img, detectors["face"])
    faces = cut_faces(img, faces_locate)
    tools.show_images(faces)
    mark_faces(img, faces_locate)

    # tools.show_images([marked_image])
    # face_images = cut_faces_from_image(img, faces_locate)
    # tools.show_images(face_images)
