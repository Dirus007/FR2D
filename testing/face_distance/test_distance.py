import os
import face_recognition
import numpy as np
from scipy.spatial import distance
import cv2
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "recording.avi")
image_folder = os.path.join(script_dir, "images")
analysis_folder = os.path.join(script_dir, "analysis")
encodings = {}


def float_to_binary(encoding, threshold=0.0):
    binary_encoding = np.where(encoding > threshold, 1, 0)
    return binary_encoding


def find_nearest_face(face_encoding, target_encoding, method):
    if method == 'euclidean':
        # L2 Norm
        #face_distances = face_recognition.face_distance([target_encoding], face_encoding)
        face_distances = distance.cdist([face_encoding], [target_encoding], 'euclidean')[0]
    elif method == 'manhattan':
        # L1 Norm
        face_distances = distance.cdist([face_encoding], [target_encoding], 'cityblock')[0]
    elif method == 'cosine':
        # Cosine Similarity
        face_distances = 1 - distance.cdist([face_encoding], [target_encoding], 'cosine')[0]
    elif method == 'mahalanobis':
        V = np.cov([target_encoding], rowvar=False)
        face_distances = distance.cdist([face_encoding], [target_encoding], 'mahalanobis', V=V)[0]
    elif method == 'hamming':
        binary_encoding = float_to_binary(face_encoding)
        binary_target_encoding = float_to_binary(target_encoding)
        face_distances = distance.cdist([binary_encoding], [binary_target_encoding], 'hamming')[0]
    else:
        return 0

    return min(face_distances) if face_distances.size > 0 else 0



for filename in os.listdir(image_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        base_filename = os.path.splitext(filename)[0]
        euclidean_distances = []
        #manhattan_distances = []
        cosine_distances = []
        #mahalanobis_distances = []
        hamming_distances = []

        cap = cv2.VideoCapture(video_path)

        print(f"Analysis started on {base_filename}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:  # Check if any faces are detected
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    euclidean_distances.append(find_nearest_face(face_encoding, encoding, 'euclidean'))
                    #manhattan_distances.append(find_nearest_face(face_encoding, encoding, 'manhattan'))
                    cosine_distances.append(find_nearest_face(face_encoding, encoding, 'cosine'))
                    #mahalanobis_distances.append(find_nearest_face(face_encoding, encoding, 'mahalanobis'))
                    hamming_distances.append(find_nearest_face(face_encoding, encoding, 'hamming'))


        cap.release()

        plt.figure(figsize=(10, 6))
        plt.plot(euclidean_distances, label='Euclidean')
        # plt.plot(manhattan_distances, label='Manhattan')
        plt.plot(cosine_distances, label='Cosine')
        # plt.plot(mahalanobis_distances, label='Mahalanobis')
        plt.plot(hamming_distances, label='Hamming')
        plt.axhline(y=0.45, color='r', linestyle='--', label='Threshold')
        plt.xlabel('Frame')
        plt.ylabel('Distance')

        plt.title(f"Distance Metrics Comparison Over Video Frames {base_filename}")
        plt.legend()
        plot_filename = os.path.join(analysis_folder, f"{base_filename}.png")
        plt.savefig(plot_filename)
        plt.show()
        print(f"Analysis finished on {base_filename}")
print("Done")
