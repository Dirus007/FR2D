import numpy as np
import cv2
import imutils
import face_recognition


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


def find_nearest_face(data, encoding, FACE_DISTANCE_THRESHOLD):
    name = "Unknown"
    sno = -1
    face_distances = face_recognition.face_distance(data["encodings"], encoding)
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < FACE_DISTANCE_THRESHOLD:
            sno = data["service_number"][best_match_index]
            name = data["names"][best_match_index]

    return sno, name