# coding: utf-8
import os
import pickle
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import logging

# извлекает кадровые ROI + превращает их в тензоры
from modalities.body.extractor import get_metadata


class MultimodalDataset(Dataset):
    """
    Мультимодальный датасет для body, face (позже — audio, text, scene).
    Читает CSV, извлекает признаки, кеширует их в pickle.
    """

    def __init__(
        self,
        csv_path,
        video_dir,
        audio_dir,
        config,
        split,
        modality_processors,
        modality_feature_extractors,
        dataset_name,
        device: str = "cuda",
    ):
        super().__init__()

        # ───────── базовые поля ─────────
        self.csv_path          = csv_path
        self.video_dir         = video_dir
        self.audio_dir         = audio_dir
        self.config            = config
        self.split             = split
        self.dataset_name      = dataset_name
        self.device            = device
        self.segment_length    = config.counter_need_frames
        self.subset_size       = config.subset_size

        # ───────── словари модальностей ─────────
        self.modality_processors         = modality_processors
        self.extractors: dict[str, object] = modality_feature_extractors

        # ───────── настройка кеша ─────────
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path  = config.save_feature_path
        self.feature_filename   = (
            f"{self.dataset_name}_{self.split}"
            f"_seed_{config.random_seed}_subset_size_{self.subset_size}"
            f"_seg_{self.segment_length}_roi_{config.roi_video}"
            f"_video_model_{config.image_model_type}_feature_norm_{config.emb_normalize}.pickle"
        )

        self.meta: list[dict] = []

        # ───── установка лейблов ─────
        if self.dataset_name == 'cmu_mosei':
            self.label_columns = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
        elif self.dataset_name == 'fiv2':
            self.label_columns = ["openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"]
        else:
            raise ValueError(f"Неизвестное имя датасета: {self.dataset_name}")

        # ───────── читаем CSV ─────────
        self.df = pd.read_csv(self.csv_path).dropna()

        if self.subset_size > 0:
            self.df = self.df.head(self.subset_size)
            logging.info(f"[DatasetMultiModal] Используем только первые {len(self.df)} записей (subset_size={self.subset_size}).")

        self.video_names = sorted(self.df["video_name"].unique())

        # ───────── либо грузим из pickle, либо готовим заново ─────────
        if self.save_prepared_data:
            os.makedirs(self.save_feature_path, exist_ok=True)
            self.pickle_path = os.path.join(self.save_feature_path, self.feature_filename)
            self.load_data(self.pickle_path)

            if not self.meta:            # pickle пуст — готовим заново
                self.prepare_data()
                self.save_data(self.pickle_path)
        else:
            self.prepare_data()

    # ──────────────────────────────────────────────────────────────────
    # служебка
    def find_file_recursive(self, base_dir: str, base_filename: str):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if os.path.splitext(file)[0] == base_filename:
                    return os.path.join(root, file)
        return None

    # ──────────────────────────────────────────────────────────────────
    # извлечение фичей
    def prepare_data(self):

        for name in tqdm(self.video_names, desc="Extracting multimodal features"):

            video_path = self.find_file_recursive(self.video_dir, name)
            audio_path = self.find_file_recursive(self.audio_dir, name)

            if video_path is None:
                print(f"❌ Видео не найдено: {name}")
                continue
            if audio_path is None:
                print(f"❌ Аудио не найдено: {name}")
                continue

            entry = {
                "sample_name": name,
                "video_path": video_path,
                "audio_path": audio_path,
                "features": {},
            }

            try:
                # --- детекция и препроцессинг кадров -----------------------
                _, body_tensor, face_tensor = get_metadata(
                    video_path      = video_path,
                    segment_length  = self.segment_length,
                    image_processor = self.modality_processors.get("body"),
                    device          = self.device,
                )

                # --- извлечение признаков через предобученные модели ------
                extracted = self.extractors["body"].extract(
                    body_tensor = body_tensor,
                    face_tensor = face_tensor,
                )

                entry["features"]["body"] = extracted.get("body")
                entry["features"]["face"] = extracted.get("face")

            except Exception as e:
                print(f"⚠️ Ошибка при извлечении фичей для {name}: {e}")
                entry["features"]["body"] = None
                entry["features"]["face"] = None

            try:
                audio_feats = self.extractors["audio"].extract(audio_path=audio_path)
                entry["features"]["audio"] = audio_feats
            except Exception as e:
                print(f"⚠️ Ошибка при извлечении аудио для {name}: {e}")
                entry["features"]["audio"] = None

            try:
                text_feats = self.extractors["text"].extract(self.df[self.df["video_name"] == name]["text"].values[0])
                entry["features"]["text"] = text_feats
            except Exception as e:
                print(f"⚠️ Ошибка при извлечении текста для {name}: {e}")
                entry["features"]["text"] = None

            # ── заглушки под будущие модальности ──

            entry["features"]["scene"] = None

            try:
                entry["label"] = torch.tensor(
                    self.df[self.df["video_name"] == name][self.label_columns].values[0],
                    dtype=torch.float32
                )
            except Exception as e:
                print(f"⚠️ Ошибка при извлечении лейблов для {name}: {e}")
                entry["label"] = torch.tensor([])

            self.meta.append(entry)
            torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────
    # работа с pickle
    def save_data(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.meta = []

    # ──────────────────────────────────────────────────────────────────
    # стандартные методы Dataset
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        return self.meta[idx]
