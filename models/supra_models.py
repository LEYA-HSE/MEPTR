import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedEmotionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class ModalityAuxEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class PersonalityRegressor(nn.Module):
    def __init__(self, input_dim, num_traits=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_traits)

    def forward(self, x):
        return self.fc(x)


class SupraMultitaskModel(nn.Module):
    def __init__(self, input_dims: dict, hidden_dim: int = 256, emo_out_dim: int = 7, pkl_out_dim: int = 5):
        super().__init__()

        self.modalities = list(input_dims.keys())

        self.shared_encoders = nn.ModuleDict({
            modality: SharedEmotionEncoder(input_dim, hidden_dim)
            for modality, input_dim in input_dims.items()
        })

        self.aux_encoders = nn.ModuleDict({
            modality: ModalityAuxEncoder(input_dim, hidden_dim)
            for modality, input_dim in input_dims.items()
        })

        self.emo_heads = nn.ModuleDict({
            modality: EmotionClassifier(hidden_dim, emo_out_dim)
            for modality in self.modalities
        })

        self.pkl_heads = nn.ModuleDict({
            modality: PersonalityRegressor(hidden_dim, pkl_out_dim)
            for modality in self.modalities
        })

    def forward(self, x_dict: dict, modality: str):
        x = x_dict[modality]  # input tensor for a single modality
        z_emo = self.shared_encoders[modality](x)
        z_aux = self.aux_encoders[modality](x)
        emo_logits = self.emo_heads[modality](z_emo)
        pkl_scores = self.pkl_heads[modality](z_aux)

        return {
            "z_emo": z_emo,
            "z_aux": z_aux,
            "emotion_logits": emo_logits,
            "personality_scores": pkl_scores
        }
