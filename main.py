# coding: utf-8
import logging
import os
import shutil
import datetime
import toml
# os.environ["HF_HOME"] = "models"
from torch.utils.data import ConcatDataset, DataLoader

from utils.config_loader import ConfigLoader
from utils.logger_setup import setup_logger
from utils.search_utils import greedy_search, exhaustive_search
from data_loading.dataset_builder import make_dataset_and_loader


from modalities.video.feature_extractor import PretrainedImageEmbeddingExtractor
from modalities.audio.feature_extractor import PretrainedAudioEmbeddingExtractor
from modalities.text.feature_extractor import PretrainedTextEmbeddingExtractor

from training.train import train as supra_train
def main():
    # ──────────────────── 1. Конфиг и директории ────────────────────
    base_config = ConfigLoader("config.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    base_config.checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(base_config.checkpoint_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # ──────────────────── 2. Логирование ────────────────────────────
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)
    base_config.show_config()

    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix     = os.path.join(epochlog_dir, "metrics_epochlog")


    # ──────────────────── 3. Процессоры + экстракторы ────────────────
    image_feature_extractor = PretrainedImageEmbeddingExtractor( device=base_config.device)
    audio_feature_extractor = PretrainedAudioEmbeddingExtractor(device=base_config.device)
    text_feature_extractor = PretrainedTextEmbeddingExtractor(device=base_config.device)

    modality_processors = {
        "body": image_feature_extractor.processor,
        "face": image_feature_extractor.processor,
        "audio": audio_feature_extractor.processor,
        "text":  None,
        "scene": image_feature_extractor.processor,
    }

    modality_extractors = {
        "body": image_feature_extractor,
        "face": image_feature_extractor,
        "audio": audio_feature_extractor,
        "text":  text_feature_extractor,
        "scene": image_feature_extractor,
    }

    # ──────────────────── 4. Даталоадеры ────────────────────────────
    train_loaders, dev_loaders, test_loaders = {}, {}, {}

    for dataset_name in base_config.datasets:
        # train
        _, train_loader = make_dataset_and_loader(
            base_config, "train",
            modality_processors, modality_extractors,
            only_dataset=dataset_name,
        )

        # dev / val
        dev_split = "dev" if os.path.exists(
            base_config.datasets[dataset_name]["csv_path"].format(
                base_dir=base_config.datasets[dataset_name]["base_dir"],
                split    ="dev",
            )
        ) else "val"

        _, dev_loader = make_dataset_and_loader(
            base_config, dev_split,
            modality_processors, modality_extractors,
            only_dataset=dataset_name,
        )

        # test
        test_split_path = base_config.datasets[dataset_name]["csv_path"].format(
            base_dir=base_config.datasets[dataset_name]["base_dir"],
            split    ="test",
        )
        if os.path.exists(test_split_path):
            _, test_loader = make_dataset_and_loader(
                base_config, "test",
                modality_processors, modality_extractors,
                only_dataset=dataset_name,
            )
        else:
            test_loader = dev_loader  # fall-back

        train_loaders[dataset_name] = train_loader
        dev_loaders[dataset_name]   = dev_loader
        test_loaders[dataset_name]  = test_loader

    # ──────────────────── 5. prepare_only ──────────────────────────
    if base_config.prepare_only:
        logging.info("== Режим prepare_only: только подготовка данных, без обучения ==")
        return

    # ──────────────────── 6. Запуск supra-modal multitask train ───────────────────
    train_datasets = []
    for ds_name in base_config.datasets:
        ds, loader = make_dataset_and_loader(
            base_config, "train",
            modality_processors, modality_extractors,
            only_dataset=ds_name
        )
        train_datasets.append(ds)

    # ───────────────────── объединяем train_datasets ──────────────────────
    union_train_ds = ConcatDataset(train_datasets)
    # возьмём collate_fn из любого из исходных лоадеров (все они одинаковые)
    sample_loader = next(iter(train_loaders.values()))
    union_train_loader = DataLoader(
        union_train_ds,
        batch_size=base_config.batch_size,
        shuffle=True,
        num_workers=base_config.num_workers,
        collate_fn=sample_loader.collate_fn
    )

    # ──────────────── запускаем supra-modal training ─────────────────────

    supra_train(cfg=base_config, mm_loader=union_train_loader)

    # # ──────────────────── 6. Поиск гиперпараметров / одиночный run ──
    # search_config  = toml.load("search_params.toml")
    # param_grid     = dict(search_config["grid"])
    # default_values = dict(search_config["defaults"])

    # if base_config.search_type == "greedy":
    #     greedy_search(
    #         base_config       = base_config,
    #         train_loader      = train_loaders,
    #         dev_loader        = dev_loaders,
    #         test_loader       = test_loaders,
    #         train_fn          = train_once,
    #         overrides_file    = overrides_file,
    #         param_grid        = param_grid,
    #         default_values    = default_values,
    #         csv_prefix        = csv_prefix,
    #         model_stage       = base_config.model_stage
    #     )

    # elif base_config.search_type == "exhaustive":
    #     exhaustive_search(
    #         base_config       = base_config,
    #         train_loader      = train_loaders,
    #         dev_loader        = dev_loaders,
    #         test_loader       = test_loaders,
    #         train_fn          = train_once,
    #         overrides_file    = overrides_file,
    #         param_grid        = param_grid,
    #         csv_prefix        = csv_prefix,
    #         model_stage       = base_config.model_stage

    #     )

    # elif base_config.search_type == "none":
    #     logging.info("== Режим одиночной тренировки (без поиска параметров) ==")

    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     csv_file_path = f"{csv_prefix}_single_{timestamp}.csv"

    #     train_once(
    #         config           = base_config,
    #         train_loader     = train_loaders,
    #         dev_loaders      = dev_loaders,
    #         test_loaders     = test_loaders,
    #         metrics_csv_path = csv_file_path,
    #         model_stage      = base_config.model_stage
    #     )

    # else:
    #     raise ValueError(f"⛔️ Неверное значение search_type в конфиге: '{base_config.search_type}'. Используй 'greedy', 'exhaustive' или 'none'.")


if __name__ == "__main__":
    main()
