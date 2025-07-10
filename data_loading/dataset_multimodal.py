# coding: utf-8
import os, pickle, logging
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

# извлекает кадровые ROI + превращает их в тензоры
from modalities.video.video_preprocessor import get_metadata


class MultimodalDataset(Dataset):
    """
    Мультимодальный датасет для body, face (позже — audio, text, scene).
    Читает CSV, извлекает признаки, кеширует их в pickle.
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        audio_dir: str,
        config,
        split: str,
        modality_processors: dict,
        modality_feature_extractors: dict,
        dataset_name: str,
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
        self.average_features  = config.average_features

        # ───────── словари модальностей ─────────
        self.modality_processors         = modality_processors
        self.extractors: dict[str, object] = modality_feature_extractors

        # ───────── настройка кеша ─────────
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path  = config.save_feature_path
        self.feature_filename   = (
            f"{self.dataset_name}_{self.split}"
            f"_seed_{config.random_seed}_subset_size_{self.subset_size}"
            f"_average_features_{self.average_features}_feature_norm_{config.emb_normalize}.pickle"
        )


        # ───── установка лейблов ─────
        if self.dataset_name == 'cmu_mosei':
            self.label_columns = [
                "Neutral", "Anger", "Disgust", "Fear",
                "Happiness", "Sadness", "Surprise"
                ]
            self.label_key = "emotion"
        elif self.dataset_name == 'fiv2':
            self.label_columns = [
                "openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"
                ]
            self.label_key = "personality"
        else:
            raise ValueError(f"Неизвестное датасет: {self.dataset_name}")

        # ───────── читаем CSV ─────────
        self.df = pd.read_csv(self.csv_path).dropna()

        if self.subset_size > 0:
            self.df = self.df.head(self.subset_size)
            logging.info(f"[DatasetMultiModal] Используем только первые {len(self.df)} записей (subset_size={self.subset_size}).")

        self.video_names = sorted(self.df["video_name"].unique())
        self.meta: list[dict] = []

        # ───────── либо грузим из pickle, либо готовим заново ─────────
        if self.save_prepared_data:
            os.makedirs(self.save_feature_path, exist_ok=True)
            self.pickle_path = os.path.join(self.save_feature_path, self.feature_filename)
            self._load_pickle(self.pickle_path)

            if not self.meta:
                self._prepare_data()
                self._save_pickle(self.pickle_path)
        else:
            self._prepare_data()

    # ────────────────────────── utils ──────────────────────────── #
    def _find_file(self, base_dir: str, base_filename: str):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if os.path.splitext(file)[0] == base_filename:
                    return os.path.join(root, file)
        return None

    def _save_pickle(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_pickle(self, filename):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                self.meta = pickle.load(f)
        else:
            self.meta = []

    def _make_label_dict(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Упаковываем метку в словарь так, как ждёт Supra pipeline."""
        return {self.label_key: tensor}

    # ──────────────────────────────────────────────────────────────────
    # извлечение фичей

    def _prepare_data(self):

        for name in tqdm(self.video_names, desc="Extracting multimodal features"):

            video_path = self._find_file(self.video_dir, name)
            audio_path = self._find_file(self.audio_dir, name)

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

            # ---------- визуальные модальности -------------------- #
            try:
                # --- детекция и препроцессинг кадров -----------------------
                __, body, face, scene = get_metadata(
                    video_path      = video_path,
                    segment_length  = self.segment_length,
                    image_processor = self.modality_processors.get("body"),
                    device          = self.device,
                )

                # --- извлечение признаков через предобученные модели ------
                extracted = self.extractors["body"].extract(
                    body_tensor = body,
                    face_tensor = face,
                    scene_tensor = scene,
                )

                for m in ("body", "face", "scene"):
                    entry["features"][m] = (
                        self._aggregate(extracted.get(m), self.average_features)
                        )
            except Exception as e:
                logging.warning(f"Video extract error {name}: {e}")

            # ---------- audio / text ------------------------------ #
            try:
                audio_feats = self.extractors["audio"].extract(audio_path=audio_path)
                entry["features"]["audio"] = self._aggregate(audio_feats, self.average_features)
            except Exception as e:
                logging.warning(f"Audio extract error {name}: {e}")
                entry["features"]["audio"] = None

            try:
                txt_raw = self.df[self.df["video_name"] == name]["text"].values[0]
                text_feats = self.extractors["text"].extract(txt_raw)
                entry["features"]["text"] = self._aggregate(text_feats, self.average_features)
            except Exception as e:
                logging.warning(f"Text extract error {name}: {e}")
                entry["features"]["text"] = None

            # ---------- label ------------------------------------- #
            try:
                lbl_tensor = torch.tensor(
                    self.df[self.df["video_name"] == name][self.label_columns].values[0],
                    dtype=torch.float32
                )
                entry["labels"] = self._make_label_dict(lbl_tensor)
            except Exception as e:
                logging.warning(f"Label extract error {name}: {e}")
                entry["labels"] = self._make_label_dict(torch.tensor([]))

            self.meta.append(entry)
            torch.cuda.empty_cache()

    def _aggregate(self, feats, average: bool = None):

        """
        Унифицированная агрегация фичей.

        Args:
            feats (Union[Tensor, dict, None]): Входные фичи.
            average (bool): Если True — усредняет по времени (dim=1), если применимо.

        Returns:
            Aggregated feats или None.

        - Если feats = Tensor с shape [B, T, D] и average=True → усредняет по T.
        - Если average=False → возвращает как есть.
        - Если feats — dict → обходит рекурсивно.
        - Если feats = None → вернёт None.
        """

        if average is None:
            average = self.average_features

        if feats is None:
            return None

        if isinstance(feats, torch.Tensor):
            if average and feats.ndim == 3:
                feats = feats.mean(dim=1)  # → [B, D]
            return feats.squeeze()

        if isinstance(feats, dict):
            return {
                key: self._aggregate(val, average)
                for key, val in feats.items()
            }

        raise TypeError(f"Unsupported feature type: {type(feats)}")


    # ───────────────────── dataset API ─────────────────────────── #
    def __len__(self):  return len(self.meta)

    def __getitem__(self, idx):
        return self.meta[idx]
