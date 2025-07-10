import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from data_loading.dataset_multimodal import MultimodalDataset


def pad_to(x, target_size):
    n_repeat = target_size - x.size(0)
    if n_repeat <= 0:
        return x
    pad = x[-1:].repeat(n_repeat, *[1 for _ in x.shape[1:]])
    return torch.cat([x, pad], dim=0)

def custom_collate_fn(batch):
    """Собирает список образцов в единый батч, отбрасывая None (невалидные)."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None

    video_path = [b["video_path"] for b in batch]

    labels = [b["label"] for b in batch]
    label_tensor = torch.stack(labels)

    videos = [b["video"] for b in batch] # new
    video_tensor = pad_sequence(videos, batch_first=True) # new

    return {
        "video_path": video_path,
        "video": video_tensor, # new
        "label": label_tensor,
    }


def make_dataset_and_loader(
    config,
    split: str,
    modality_processors: dict,
    modality_extractors: dict,
    only_dataset: str = None,
):
    """
    Собирает MultimodalDataset и возвращает DataLoader.
    modality_processors — dict с CLIPProcessor-ами и т.п.
    modality_extractors  — dict с предобученными моделями.
    """
    datasets = []

    if not getattr(config, "datasets", None):
        raise ValueError("⛔ В конфиге не указана секция [datasets].")

    for dataset_name, dataset_cfg in config.datasets.items():
        if only_dataset and dataset_name != only_dataset:
            continue

        csv_path = dataset_cfg["csv_path"].format(
            base_dir=dataset_cfg["base_dir"],
            split=split,
        )
        video_dir = dataset_cfg["video_dir"].format(
            base_dir=dataset_cfg["base_dir"],
            split=split,
        )

        audio_dir = dataset_cfg["audio_dir"].format(
            base_dir=dataset_cfg["base_dir"],
            split=split,
        )

        dataset = MultimodalDataset(
            csv_path=csv_path,
            video_dir=video_dir,
            audio_dir=audio_dir,
            config=config,
            split=split,
            modality_processors=modality_processors,
            modality_feature_extractors=modality_extractors,
            dataset_name=dataset_name,
            device=config.device,
        )
        datasets.append(dataset)

    if not datasets:
        raise ValueError(f"⚠️ Для split='{split}' не найдено ни одного датасета.")

    full_dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)

    loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        collate_fn=custom_collate_fn,
    )

    return full_dataset, loader
