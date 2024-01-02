import numpy as np
import cv2
import imutils
import face_recognition
from scipy.spatial import distance


def float_to_binary(encoding, threshold=0.0):
    binary_encoding = np.where(encoding > threshold, 1, 0)
    return binary_encoding


def get_available_cameras():
    camera_list = []
    i = 0
    while True:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            break
        camera_list.append(i)
        cap.release()
        i += 1
    return camera_list


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_all_faces(frame, FRAME_WIDTH):
    frame = imutils.resize(frame, width=FRAME_WIDTH)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    return face_locations, rgb_frame, frame


def find_most_prominent_face(face_locations):
    largest_face_area = 0
    largest_face_location = None

    # Identify the most prominent face only
    for (top, right, bottom, left) in face_locations:
        face_area = (bottom - top) * (right - left)
        if face_area > largest_face_area:
            largest_face_area = face_area
            largest_face_location = (top, right, bottom, left)

    return largest_face_location


def process_eyes(frame, face_landmarks):
    for eye in ['left_eye', 'right_eye']:
        for point in face_landmarks[eye]:
            cv2.circle(frame, point, 1, (255, 255, 0), -1)

    leftEye = np.array(face_landmarks['left_eye'])
    rightEye = np.array(face_landmarks['right_eye'])
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0

    return ear


def find_nearest_face(data, encoding, FACE_DISTANCE_THRESHOLD, method):
    name = "Unknown"
    sno = -1
    face_distances = []
    if method == 'euclidean':
        # L2 Norm
        face_distances = face_recognition.face_distance(data["encodings"], encoding)
    elif method == 'manhattan':
        # L1 Norm
        face_distances = distance.cdist([encoding], data["encodings"], 'cityblock')[0]
    elif method == 'cosine':
        # Cosine Similarity
        face_distances = 1 - distance.cdist([encoding], data["encodings"], 'cosine')[0]

    elif method == 'mahalanobis':
        V = np.cov(data["encodings"], rowvar=False)
        face_distances = distance.cdist([encoding], data["encodings"], 'mahalanobis', V=V)[0]
    elif method == 'hamming':
        binary_encoding = float_to_binary(encoding)
        binary_encodings_data = np.array([float_to_binary(enc) for enc in data["encodings"]])
        face_distances = distance.cdist([binary_encoding], binary_encodings_data, 'hamming')[0]

    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < FACE_DISTANCE_THRESHOLD:
            sno = data["service_number"][best_match_index]
            name = data["names"][best_match_index]

    return sno, name


def landmarks_list_and_encodings(rgb_frame, largest_face_location):
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame, [largest_face_location])
    face_encodings = face_recognition.face_encodings(rgb_frame, [largest_face_location])

    return face_landmarks_list, face_encodings


def zoom_frame(frame, zoom_level):
    original_height, original_width = frame.shape[:2]
    if zoom_level != 100:
        scale_factor = zoom_level / 100
        new_width = int(original_width / scale_factor)
        new_height = int(original_height / scale_factor)
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2
        frame = frame[top:top + new_height, left:left + new_width]
        frame = cv2.resize(frame, (original_width, original_height))

    return frame
