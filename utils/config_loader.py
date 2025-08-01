# utils/config_loader.py

import os
import toml
import logging

class ConfigLoader:
    """
    Класс для загрузки и обработки конфигурации из `config.toml`.
    """

    def __init__(self, config_path="config.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации `{config_path}` не найден!")

        self.config = toml.load(config_path)

        # ---------------------------
        # Общие параметры
        # ---------------------------
        self.split = self.config.get("split", "train")

        # ---------------------------
        # Пути к данным
        # ---------------------------
        self.datasets = self.config.get("datasets", {})

        # ---------------------------
        # Пути к синтетическим данным
        # ---------------------------
        synthetic_data_cfg = self.config.get("synthetic_data", {})
        self.use_synthetic_data = synthetic_data_cfg.get("use_synthetic_data", False)
        self.synthetic_path = synthetic_data_cfg.get("synthetic_path", "E:/MELD_S")
        self.synthetic_ratio = synthetic_data_cfg.get("synthetic_ratio", 0.0)

        # ---------------------------
        # Модальности и эмоции
        # ---------------------------
        self.modalities = self.config.get("modalities", ["audio"])
        self.emotion_columns = self.config.get("emotion_columns", ["Neutral","Anger","Disgust","Fear","Happiness","Sadness","Surprise","Other"])
        # ---------------------------
        # DataLoader
        # ---------------------------
        dataloader_cfg = self.config.get("dataloader", {})
        self.num_workers = dataloader_cfg.get("num_workers", 0)
        self.shuffle = dataloader_cfg.get("shuffle", True)
        self.prepare_only = dataloader_cfg.get("prepare_only", False)
        self.average_features = dataloader_cfg.get("average_features", False)

        # ---------------------------
        # Тренировка: общие
        # ---------------------------
        train_general = self.config.get("train", {}).get("general", {})
        self.random_seed = train_general.get("random_seed", 42)
        self.subset_size = train_general.get("subset_size", 0)
        self.merge_probability = train_general.get("merge_probability", 0)
        self.batch_size = train_general.get("batch_size", 8)
        self.num_epochs = train_general.get("num_epochs", 100)
        self.max_patience = train_general.get("max_patience", 10)
        self.save_best_model = train_general.get("save_best_model", False)
        self.save_prepared_data = train_general.get("save_prepared_data", True)
        self.save_feature_path = train_general.get("save_feature_path", "./features/")
        self.search_type = train_general.get("search_type", "none")
        self.smoothing_probability = train_general.get("smoothing_probability", 0)
        self.path_to_df_ls = train_general.get("path_to_df_ls", None)
        self.early_stop_on = train_general.get("early_stop_on", "dev")
        self.lambda_emotion = train_general.get("lambda_emotion", 1)
        self.lambda_personality = train_general.get("lambda_personality", 5)
        self.lambda_domain = train_general.get("lambda_domain", 0.1)
        self.checkpoint_dir = train_general.get("checkpoint_dir","checkpoints")
        self.device = train_general.get("device", "cuda")
        self.selection_metric = train_general.get("selection_metric", "mean_combo")
        self.single_task = train_general.get("single_task", False)

        # ---------------------------
        # Тренировка: параметры модели
        # ---------------------------
        train_model = self.config.get("train", {}).get("model", {})
        self.id_ablation_type_by_modality = train_model.get("id_ablation_type_by_modality", 0)
        self.id_ablation_type_by_component = train_model.get("id_ablation_type_by_component", 6)
        self.single_task_id = train_model.get("single_task_id", 0)
        self.model_name = train_model.get("model_name", "BiFormer")
        self.model_stage = train_model.get("model_stage", "emotion")
        self.path_to_saved_emotion_model = train_model.get("path_to_saved_emotion_model", None)
        self.path_to_saved_personality_model = train_model.get("path_to_saved_personality_model", None)
        self.per_activation = train_model.get("per_activation", "sigmoid")
        self.weight_emotion = train_model.get("weight_emotion", 0.1)
        self.weight_pers = train_model.get("weight_pers", 1)
        self.pers_loss_type = train_model.get("pers_loss_type", True)
        self.emotion_loss_type = train_model.get("emotion_loss_type", True)
        self.flag_emo_weight = train_model.get("flag_emo_weight", "ccc")
        self.hidden_dim = train_model.get("hidden_dim", 256)
        self.hidden_dim_gated = train_model.get("hidden_dim_gated", 256)
        self.num_transformer_heads = train_model.get("num_transformer_heads", 8)
        self.num_graph_heads = train_model.get("num_graph_heads", 8)
        self.tr_layer_number = train_model.get("tr_layer_number", 1)
        self.mamba_d_state = train_model.get("mamba_d_state", 16)
        self.mamba_ker_size = train_model.get("mamba_ker_size", 4)
        self.mamba_layer_number = train_model.get("mamba_layer_number", 3)
        self.positional_encoding = train_model.get("positional_encoding", True)
        self.dropout = train_model.get("dropout", 0.0)
        self.out_features = train_model.get("out_features", 128)
        self.mode = train_model.get("mode", "mean")
        self.fusion_dim = train_model.get("fusion_dim", 64)

        # параметры для лучших эмоциональной и персональной моделей
        self.hidden_dim_emo = train_model.get("hidden_dim_emo", 256)
        self.out_features_emo = train_model.get("out_features_emo", 256)
        self.name_best_emo_model = train_model.get("name_best_emo_model", "BiFormer")
        self.name_best_per_model = train_model.get("name_best_per_model", "BiFormer")
        self.path_to_saved_emotion_model = train_model.get("path_to_saved_emotion_model", None)
        self.path_to_saved_personality_model = train_model.get("path_to_saved_personality_model", None)
        self.num_transformer_heads_emo = train_model.get("num_transformer_heads_emo", 8)
        self.tr_layer_number_emo = train_model.get("tr_layer_number_emo", 1)
        self.positional_encoding_emo = train_model.get("positional_encoding_emo", True)
        self.mamba_d_state_emo = train_model.get("mamba_d_state_emo", 16)
        self.mamba_layer_number_emo = train_model.get("mamba_layer_number_emo", 3)
        self.hidden_dim_per = train_model.get("hidden_dim_per", 256)
        self.out_features_per = train_model.get("out_features_per", 256)
        self.num_transformer_heads_per = train_model.get("num_transformer_heads_per", 8)
        self.tr_layer_number_per = train_model.get("tr_layer_number_per", 1)
        self.positional_encoding_per = train_model.get("positional_encoding_per", True)
        self.mamba_d_state_per = train_model.get("mamba_d_state_per", 16)
        self.mamba_layer_number_per = train_model.get("mamba_layer_number_per", 3)
        self.best_per_activation = train_model.get("best_per_activation", "sigmoid")

        # ---------------------------
        # Тренировка: оптимизатор
        # ---------------------------
        train_optimizer = self.config.get("train", {}).get("optimizer", {})
        self.optimizer = train_optimizer.get("optimizer", "adam")
        self.lr = train_optimizer.get("lr", 1e-4)
        self.weight_decay = train_optimizer.get("weight_decay", 0.0)
        self.momentum = train_optimizer.get("momentum", 0.9)

        # ---------------------------
        # Тренировка: шедулер
        # ---------------------------
        train_scheduler = self.config.get("train", {}).get("scheduler", {})
        self.scheduler_type = train_scheduler.get("scheduler_type", "plateau")
        self.warmup_ratio = train_scheduler.get("warmup_ratio", 0.1)

        # ---------------------------
        # Эмбеддинги
        # ---------------------------
        emb_cfg = self.config.get("embeddings", {})
        self.audio_model_name = emb_cfg.get("audio_model", "amiriparian/ExHuBERT")
        self.text_model_name  = emb_cfg.get("text_model", "jinaai/jina-embeddings-v3")
        self.audio_classifier_checkpoint = emb_cfg.get("audio_classifier_checkpoint", "best_audio_model.pt")
        self.text_classifier_checkpoint = emb_cfg.get("text_classifier_checkpoint", "best_text_model.pth")
        self.image_classifier_checkpoint = emb_cfg.get("image_classifier_checkpoint", "torchscript_model_0_66_37_wo_gl.pth")
        self.image_model_type = emb_cfg.get("image_model_type", "resnet50")
        self.image_embedding_dim = emb_cfg.get("image_embedding_dim", 512)
        self.cut_target_layer = emb_cfg.get("cut_target_layer", 2)
        self.roi_video = emb_cfg.get("roi_video", "face")
        self.counter_need_frames = emb_cfg.get("counter_need_frames", 20)
        self.image_size = emb_cfg.get("image_size", 224)
        self.audio_embedding_dim = emb_cfg.get("audio_embedding_dim", 1024)
        self.text_embedding_dim  = emb_cfg.get("text_embedding_dim", 1024)
        self.emb_normalize = emb_cfg.get("emb_normalize", True)
        self.audio_pooling = emb_cfg.get("audio_pooling", None)
        self.text_pooling  = emb_cfg.get("text_pooling", None)
        self.max_tokens = emb_cfg.get("max_tokens", 256)
        self.window_size = emb_cfg.get("window_size", 5)

        if __name__ == "__main__":
            self.log_config()

    def log_config(self):
        logging.info("=== CONFIGURATION ===")
        logging.info(f"Split: {self.split}")
        logging.info(f"Datasets loaded: {list(self.datasets.keys())}")
        for name, ds in self.datasets.items():
            logging.info(f"[Dataset: {name}]")
            logging.info(f"  Base Dir: {ds.get('base_dir', 'N/A')}")
            logging.info(f"  CSV Path: {ds.get('csv_path', '')}")
            logging.info(f"  WAV Dir: {ds.get('wav_dir', 'N/A')}")
            logging.info(f"  Video Dir: {ds.get('video_dir', '')}")
            logging.info(f"  Audio Dir: {ds.get('audio_dir', '')}")

        # Логируем обучающие параметры
        logging.info("--- Training Config ---")
        logging.info(f"DataLoader: batch_size={self.batch_size}, num_workers={self.num_workers}, shuffle={self.shuffle}")
        logging.info(f"Model Name: {self.model_name}")
        logging.info(f"Random Seed: {self.random_seed}")
        logging.info(f"Hidden Dim: {self.hidden_dim}")
        logging.info(f"Hidden Dim in Gated: {self.hidden_dim_gated}")
        logging.info(f"Num Heads in Transformer: {self.num_transformer_heads}")
        logging.info(f"Num Heads in Graph: {self.num_graph_heads}")
        logging.info(f"Mode stat pooling: {self.mode}")
        logging.info(f"Optimizer: {self.optimizer}")
        logging.info(f"Scheduler Type: {self.scheduler_type}")
        logging.info(f"Warmup Ratio: {self.warmup_ratio}")
        logging.info(f"Weight Decay for Adam: {self.weight_decay}")
        logging.info(f"Momentum (SGD): {self.momentum}")
        logging.info(f"Positional Encoding: {self.positional_encoding}")
        logging.info(f"Number of Transformer Layers: {self.tr_layer_number}")
        logging.info(f"Mamba D State: {self.mamba_d_state}")
        logging.info(f"Mamba Kernel Size: {self.mamba_ker_size}")
        logging.info(f"Mamba Layer Number: {self.mamba_layer_number}")
        logging.info(f"Dropout: {self.dropout}")
        logging.info(f"Out Features: {self.out_features}")
        logging.info(f"LR: {self.lr}")
        logging.info(f"Num Epochs: {self.num_epochs}")
        logging.info(f"Merge Probability={self.merge_probability}")
        logging.info(f"Smoothing Probability={self.smoothing_probability}")
        logging.info(f"Max Patience={self.max_patience}")
        logging.info(f"Save Prepared Data={self.save_prepared_data}")
        logging.info(f"Path to Save Features={self.save_feature_path}")
        logging.info(f"Search Type={self.search_type}")

        # Логируем embeddings
        logging.info("--- Embeddings Config ---")
        logging.info(f"Audio Model: {self.audio_model_name}, Text Model: {self.text_model_name}")
        logging.info(f"Audio dim={self.audio_embedding_dim}, Text dim={self.text_embedding_dim}")
        logging.info(f"Audio pooling={self.audio_pooling}, Text pooling={self.text_pooling}")
        logging.info(f"Device={self.device}, Normalize={self.emb_normalize}")

    def show_config(self):
        self.log_config()
