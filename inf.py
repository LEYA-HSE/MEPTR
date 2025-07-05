# %%
import mediapipe as mp
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor
from models.models import EmotionMamba, PersonalityMamba, FusionTransformer
from data_loading.feature_extractor import PretrainedImageEmbeddingExtractor
from utils.config_loader import ConfigLoader

def draw_box(image, box, color=(255, 0, 255)):
    """Draw a rectangle on the image."""
    line_width = 2
    lw = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

def image_processing(image, image_processor):
    image = image_processor(images=image, return_tensors="pt").to("cuda")
    image = image['pixel_values']
    return image

def preprocess_face(face_roi: np.ndarray) -> np.ndarray:
    """Предобработка области лица (пример: нормализация + resize)."""
    # Пример: преобразуем в 112x112 и нормализуем [0, 1]
    face_roi = cv2.resize(face_roi, (112, 112))
    face_roi = face_roi.astype('float32') / 255.0
    return face_roi

def preprocess_body(body_roi: np.ndarray) -> np.ndarray:
    """Предобработка области тела (пример: нормализация + resize)."""
    # Пример: преобразуем в 224x224 и нормализуем [0, 1]
    body_roi = cv2.resize(body_roi, (224, 224))
    body_roi = body_roi.astype('float32') / 255.0
    return body_roi

def select_uniform_frames(frames, N):
    if len(frames) <= N:
        return frames
    else:
        indices = np.linspace(0, len(frames) - 1, num=N, dtype=int)
        return [frames[i] for i in indices]

def get_fusion_model(config, device):
    emo_model = EmotionMamba(
    input_dim_emotion     = config.image_embedding_dim,
    input_dim_personality = config.image_embedding_dim,
    len_seq               = config.counter_need_frames,
    hidden_dim            = config.hidden_dim_emo,
    out_features          = config.out_features_emo,
    tr_layer_number       = config.tr_layer_number_emo,
    num_transformer_heads = config.num_transformer_heads_emo,
    positional_encoding   = config.positional_encoding_emo,
    mamba_d_model         = config.mamba_d_state_emo,
    mamba_layer_number    = config.mamba_layer_number_emo,
    dropout               = config.dropout,
    num_emotions          = 7,
    num_traits            = 5,
    device                = device
    ).to(device).eval()
    # параметры задаем для лучшей персональной модели
    per_model = PersonalityMamba(
    input_dim_emotion     = config.image_embedding_dim,
    input_dim_personality = config.image_embedding_dim,
    len_seq               = config.counter_need_frames,
    hidden_dim            = config.hidden_dim_per,
    out_features          = config.out_features_per,
    per_activation        = config.best_per_activation,
    tr_layer_number       = config.tr_layer_number_per,
    num_transformer_heads = config.num_transformer_heads_per,
    positional_encoding   = config.positional_encoding_per,
    mamba_d_model         = config.mamba_d_state_per,
    mamba_layer_number    = config.mamba_layer_number_per,
    dropout               = config.dropout,
    num_emotions          = 7,
    num_traits            = 5,
    device                = device
    ).to(device).eval()

    # emo_state = torch.load(config.path_to_saved_emotion_model, map_location=device)
    # emo_model.load_state_dict(emo_state)

    # emo_state = torch.load(config.path_to_saved_personality_model, map_location=device)
    # per_model.load_state_dict(emo_state)
    model = FusionTransformer(
        emo_model             = emo_model,
        per_model             = per_model,
        input_dim_emotion     = config.image_embedding_dim,
        input_dim_personality = config.image_embedding_dim,
        hidden_dim            = config.hidden_dim,
        out_features          = config.out_features,
        per_activation        = config.per_activation,
        tr_layer_number       = config.tr_layer_number,
        num_transformer_heads = config.num_transformer_heads,
        positional_encoding   = config.positional_encoding,
        mamba_d_model         = config.mamba_d_state,
        mamba_layer_number    = config.mamba_layer_number,
        dropout               = config.dropout,
        num_emotions          = 7,
        num_traits            = 5,
        device                = device
        ).to(device).eval()

    return model

def transform_matrix(matrix):
    threshold1 = 1 - 1/7
    threshold2 = 1/7
    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result

def process_predictions(pred_emo):
    pred_emo = torch.nn.functional.softmax(pred_emo, dim=1).cpu().detach().numpy()
    pred_emo = transform_matrix(pred_emo).tolist()
    return pred_emo

def get_metadata(video_path: str, segment_length: int, image_processor: None, image_feature_extractor: None, device: None) -> pd.DataFrame:
    """Основная функция: получает метаданные для видео."""
    if hasattr(body_detector.predictor, 'trackers'):
        body_detector.predictor.trackers[0].reset()

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)
    w, h, fps, total_frames = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))
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
            # Детекция всех лиц
            preprocessed_body = []
            preprocessed_face = []
            face_results = face_detector.process(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            # Детекция всех тел
            body_results = body_detector.track(im0, persist=True, imgsz=640, conf=0.01, iou=0.5,
                                             augment=False, device=0, verbose=False)

            # Случай 1: Есть лица — обрабатываем каждое
            if face_results.detections:
                for face_idx, detection in enumerate(face_results.detections):
                    # Координаты лица
                    bbox = detection.location_data.relative_bounding_box
                    x1, y1 = max(int(bbox.xmin * w), 0), max(int(bbox.ymin * h), 0)
                    x2, y2 = min(int((bbox.xmin + bbox.width) * w), w), min(int((bbox.ymin + bbox.height) * h), h)
                    face_bbox = (x1, y1, x2, y2)
                    face_center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Ищем тело, содержащее центр лица
                    body_bbox = None
                    body_id = -1
                    if body_results and len(body_results[0].boxes) > 0:
                        for box in body_results[0].boxes:
                            box_coords = box.xyxy.int().cpu().numpy()[0]
                            if (box_coords[0] <= face_center[0] <= box_coords[2] and
                                box_coords[1] <= face_center[1] <= box_coords[3]):
                                body_bbox = box_coords
                                body_id = box.id.int().cpu().item() if box.id else -1
                                break

                    # Предобработка
                    face_roi = im0[y1:y2, x1:x2]
                    draw_box(im0, [x1, y1, x2, y2])
                    draw_box(im0, [body_bbox[0], body_bbox[1], body_bbox[2], body_bbox[3]])
                    preprocessed_face = image_processing(face_roi, image_processor) if face_roi.size > 0 else None

                    if body_bbox is not None:
                        body_roi = im0[body_bbox[1]:body_bbox[3], body_bbox[0]:body_bbox[2]]
                        preprocessed_body = image_processing(body_roi, image_processor) if body_roi.size > 0 else None
                    else:
                        preprocessed_body = []

                    # Сохраняем результат
                    embeds.append([
                        video_name, counter, body_id,
                        x1, y1, x2, y2,
                        body_bbox[0] if body_bbox is not None else None,
                        body_bbox[1] if body_bbox is not None else None,
                        body_bbox[2] if body_bbox is not None else None,
                        body_bbox[3] if body_bbox is not None else None,
                        # preprocessed_face,
                        # preprocessed_body
                    ])
                    # print(preprocessed_body.shape)
                    # print(preprocessed_face.shape)
                    if preprocessed_body.shape[0] > 0:
                        body_list.append(preprocessed_body)
                    if preprocessed_face.shape[0] > 0:
                        face_list.append(preprocessed_face)


            # Случай 2: Лиц нет — берём самое большое тело
            elif body_results and len(body_results[0].boxes) > 0:
                largest_body = max(
                    body_results[0].boxes,
                    key=lambda box: (box.xyxy[0,2] - box.xyxy[0,0]) * (box.xyxy[0,3] - box.xyxy[0,1])
                )
                body_coords = largest_body.xyxy.int().cpu().numpy()[0]
                body_id = largest_body.id.int().cpu().item() if largest_body.id else -1

                # Предобработка тела
                body_roi = im0[body_coords[1]:body_coords[3], body_coords[0]:body_coords[2]]
                preprocessed_body = preprocess_body(body_roi) if body_roi.size > 0 else []

                embeds.append([
                    video_name, counter, body_id,
                    None, None, None, None,  # Нет лица
                    body_coords[0], body_coords[1], body_coords[2], body_coords[3],
                    # None,  # Нет лица
                    # preprocessed_body
                ])

                if preprocessed_body.shape[0] > 0:
                    body_list.append(preprocessed_body)
                if preprocessed_face.shape[0] > 0:
                    face_list.append(preprocessed_face)

            plt.imshow(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            plt.show()

        counter += 1
        torch.cuda.empty_cache()

    cap.release()

    body_list = torch.cat(body_list, dim=0)
    body_feature = image_feature_extractor.extract(body_list).to(device)

    face_list = torch.cat(face_list, dim=0)
    face_feature = image_feature_extractor.extract(face_list).to(device)

    df = pd.DataFrame(embeds, columns=[
        "video_name", "frame", "person_id",
        "face_x1", "face_y1", "face_x2", "face_y2",
        "body_x1", "body_y1", "body_x2", "body_y2",
        # "preprocessed_face", "preprocessed_body"
    ])
    return df, body_feature, face_feature

# %%
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
body_detector = YOLO('best.pt')
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
config_body = ConfigLoader("inference_config_body.toml")
config_face = ConfigLoader("inference_config_face.toml")
image_feature_extractor = PretrainedImageEmbeddingExtractor(config_body)

# Models can download from https://drive.google.com/drive/folders/1APMtC4LXjuW9behd2TxVXz0DsjQKAgRR?usp=sharing

body_model = get_fusion_model(config_body, 'cuda')
face_model = get_fusion_model(config_face, 'cuda')
# results_clip_body_true_mamba_fusiontransformer_2025-06-27_16-10-57/metrics_by_epoch/metrics_epochlog_FusionTransformer_num_transformer_heads_16_20250627_183039_timestamp/best_model_dev.pt
body_fusion_model_path = 'clip_body_mamba_transformer_fusion_model.pt'
# results_fusiontransformer_2025-07-03_09-41-13/metrics_by_epoch/metrics_epochlog_FusionTransformer_tr_layer_number_3_20250703_124848_timestamp/best_model_dev.pt
face_fusion_model_path = 'clip_face_mamba_transformer_fusion_model.pt'

body_state = torch.load(body_fusion_model_path, map_location='cuda')
body_model.load_state_dict(body_state)

face_state = torch.load(face_fusion_model_path, map_location='cuda')
face_model.load_state_dict(face_state)

# %%
# video_path = 'E:/Databases/FirstImpressionsV2/test/test80_13/_plk5k7PBEg.003.mp4'
# segment_length = 30
# df, body_feature, face_feature = get_metadata(video_path=video_path,
#                                  segment_length=segment_length,
#                                  image_processor=image_processor,
#                                  image_feature_extractor=image_feature_extractor,
#                                  device='cuda')
# with torch.no_grad():
#     body_logits = body_model(emotion_input=torch.unsqueeze(body_feature, 0), personality_input=torch.unsqueeze(body_feature, 0))
#     face_logits = face_model(emotion_input=torch.unsqueeze(face_feature, 0), personality_input=torch.unsqueeze(face_feature, 0))
# print('Body emotion predictions: ', process_predictions(body_logits['emotion_logits']), 'Body personality predictions: ', body_logits['personality_scores'].cpu().detach().numpy())
# print('Face emotion predictions: ', process_predictions(face_logits['emotion_logits']), 'Face personality predictions: ', face_logits['personality_scores'].cpu().detach().numpy())
