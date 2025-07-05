import cv2
import os
import torch
from ultralytics import YOLO
import mediapipe as mp

from utils.body.utils import (
    select_uniform_frames,
    image_processing
)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
body_detector = YOLO('extractors/body/best.pt')

def get_metadata(video_path: str, segment_length: int, image_processor: None, device: None):
    """Основная функция: вытаскивает preprocessed body и face тензоры"""
    if hasattr(body_detector.predictor, 'trackers'):
        body_detector.predictor.trackers[0].reset()

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    w, h, fps, total_frames = (int(cap.get(x)) for x in  (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))
    need_frames = select_uniform_frames(list(range(total_frames)), segment_length)

    counter = 0
    embeds = []

    body_list = []
    face_list = []

    while True:
        ret, im0 = cap.read()
        if not ret:
            break

        if counter in need_frames:
            preprocessed_body = []
            preprocessed_face = []
            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            face_results = face_detector.process(im0)
            body_results = body_detector.track(im0, persist=True, imgsz=640, conf=0.01, iou=0.5, augment=False, device=0, verbose=False)

            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1, y1 = max(int(bbox.xmin * w), 0), max(int(bbox.ymin * h), 0)
                    x2, y2 = min(int((bbox.xmin + bbox.width) * w), w), min(int((bbox.ymin + bbox.height) * h), h)
                    face_bbox = (x1, y1, x2, y2)
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    body_bbox = None
                    if body_results and len(body_results[0].boxes) > 0:
                        for box in body_results[0].boxes:
                            box_coords = box.xyxy.int().cpu().numpy()[0]
                            if (box_coords[0] <= face_center[0] <= box_coords[2] and
                                box_coords[1] <= face_center[1] <= box_coords[3]):
                                body_bbox = box_coords
                                break

                    face_roi = im0[y1:y2, x1:x2]
                    preprocessed_face = image_processing(face_roi, image_processor) if face_roi.size > 0 else None

                    if body_bbox is not None:
                        body_roi = im0[body_bbox[1]:body_bbox[3], body_bbox[0]:body_bbox[2]]
                        preprocessed_body = image_processing(body_roi, image_processor) if body_roi.size > 0 else None

                    if preprocessed_body is not None:
                        body_list.append(preprocessed_body)
                    if preprocessed_face is not None:
                        face_list.append(preprocessed_face)

            counter += 1
            torch.cuda.empty_cache()

    cap.release()

    body_tensor = torch.cat(body_list, dim=0) if body_list else None
    face_tensor = torch.cat(face_list, dim=0) if face_list else None

    return video_name, body_tensor, face_tensor
