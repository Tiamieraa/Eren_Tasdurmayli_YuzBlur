from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2

def apply_mosaic(image, x1, y1, x2, y2, mosaic_strength=15):
    h, w, _ = image.shape
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    face_roi = image[y1:y2, x1:x2]

    if face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        return image

    small_roi = cv2.resize(face_roi, (face_roi.shape[1] // mosaic_strength, face_roi.shape[0] // mosaic_strength), interpolation=cv2.INTER_LINEAR)
    mosaic_roi = cv2.resize(small_roi, (face_roi.shape[1], face_roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    image[y1:y2, x1:x2] = mosaic_roi
    return image


def draw_landmarks_on_image(rgb_image, detection_result, apply_mosaic_effect=True, mosaic_strength=15):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        x_coords = [landmark.x * rgb_image.shape[1] for landmark in face_landmarks]
        y_coords = [landmark.y * rgb_image.shape[0] for landmark in face_landmarks]

        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))

        if apply_mosaic_effect:
            annotated_image = apply_mosaic(annotated_image, x_min, y_min, x_max, y_max, mosaic_strength)

        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        koordinatlar= []
        for landmark in face_landmarks:
            koordinatlar.append(str(round(landmark.x, 4)))
            koordinatlar.append(str(round(landmark.y,4)))
        
        koordinatlar = ",".join(koordinatlar)
        koordinatlar += f",{etiket}\n"
        with open("veriseti.csv", "a") as f:
            f.write(koordinatlar)

    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def sutun_basliklarini_olustur():
    with open("veriseti.csv", "w") as f:
        satir = ""
        for i in range(1, 479):
            satir = satir + f"x{i},y{i},"
        satir = satir + "Etiket\n"
        f.write(satir)

etiket = "happy" 
sutun_basliklarini_olustur()


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = detector.detect(mp_image)

        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result, apply_mosaic_effect=True, mosaic_strength=15)
        cv2.imshow("yuz", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            exit(0)