# coding: utf-8
import cv2
import os
import torch
from ultralytics import YOLO
import mediapipe as mp

from utils.body.utils import (
    select_uniform_frames,
    image_processing,          # использует CLIP-processor
)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.6
)

# веса YOLO лежат в extractors/body/best.pt
body_detector = YOLO("extractors/body/best.pt")


def get_metadata(
    video_path: str,
    segment_length: int,
    image_processor,              # CLIPProcessor
    device: str = "cuda",         # пока не используется, но оставим на будущее
):
    """
    Возвращает:
        video_name (str),
        body_tensor [N, 3, H, W] or None,
        face_tensor [M, 3, H, W] or None
    """
    # сбрасываем трекер YOLO, чтобы он не тянул ID между разными видео
    if hasattr(body_detector.predictor, "trackers"):
        body_detector.predictor.trackers[0].reset()

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    w, h, *_ = (int(cap.get(x)) for x in (
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_FRAME_COUNT)
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    need_frames  = select_uniform_frames(list(range(total_frames)), segment_length)

    body_list, face_list = [], []
    counter = 0

    while True:
        ret, im0 = cap.read()
        if not ret:
            break

        if counter in need_frames:
            im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            # ── детекции ───────────────────────────────────────────────
            face_results = face_detector.process(im_rgb)
            body_results = body_detector.track(
                im_rgb, persist=True, imgsz=640, conf=0.01, iou=0.5,
                augment=False, device=0, verbose=False
            )

            # ── 1) есть лица  ──────────────────────────────────────────
            if face_results.detections:
                for det in face_results.detections:
                    bbox = det.location_data.relative_bounding_box
                    x1, y1 = max(int(bbox.xmin * w), 0), max(int(bbox.ymin * h), 0)
                    x2, y2 = min(int((bbox.xmin + bbox.width) * w), w), min(int((bbox.ymin + bbox.height) * h), h)
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # ищем body-box, внутри которого лежит центр лица
                    body_bbox = None
                    if body_results and len(body_results[0].boxes):
                        for box in body_results[0].boxes:
                            bx = box.xyxy.int().cpu().numpy()[0]
                            if bx[0] <= face_center[0] <= bx[2] and bx[1] <= face_center[1] <= bx[3]:
                                body_bbox = bx
                                break

                    # ROI лица
                    face_roi = im_rgb[y1:y2, x1:x2]
                    if face_roi.size:
                        face_list.append(image_processing(face_roi, image_processor))

                    # ROI тела (если нашлось)
                    if body_bbox is not None:
                        b = body_bbox
                        body_roi = im_rgb[b[1]:b[3], b[0]:b[2]]
                        if body_roi.size:
                            body_list.append(image_processing(body_roi, image_processor))

            # ── 2) лиц нет, но YOLO нашёл тела ────────────────────────
            elif body_results and len(body_results[0].boxes):
                # берём самое крупное тело
                largest = max(
                    body_results[0].boxes,
                    key=lambda b: (b.xyxy[0, 2] - b.xyxy[0, 0]) *
                                  (b.xyxy[0, 3] - b.xyxy[0, 1])
                )
                bx = largest.xyxy.int().cpu().numpy()[0]
                body_roi = im_rgb[bx[1]:bx[3], bx[0]:bx[2]]
                if body_roi.size:
                    body_list.append(image_processing(body_roi, image_processor))

        counter += 1

    cap.release()

    body_tensor = torch.cat(body_list, dim=0) if body_list else None
    face_tensor = torch.cat(face_list, dim=0) if face_list else None
    return video_name, body_tensor, face_tensor
