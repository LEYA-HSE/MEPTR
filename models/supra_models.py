import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── helpers ──────────────────────────
class ModalityProjector(nn.Module):
    """D_m  →  shared_in_dim"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(x)


class SharedEmotionEncoder(nn.Module):
    """Один на все модальности (vmPFC-заменитель)."""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class ModalityAuxEncoder(nn.Module):
    """Личная «не-эмоция» для модальности."""
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class EmotionClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class PersonalityRegressor(nn.Module):
    def __init__(self, input_dim: int, num_traits: int = 5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_traits)

    def forward(self, x):
        return self.fc(x)


# ───────────────────── Supra-model ──────────────────────────
class SupraMultitaskModel(nn.Module):
    """
    • один shared-emotion-encoder
    • per-modality aux-encoder + PKL-голова
    """
    def __init__(
        self,
        input_dims: dict[str, int],          # {modality: embedding_dim}
        shared_in_dim: int = 512,            # куда проецируем всё
        hidden_dim: int = 256,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
    ):
        super().__init__()

        self.modalities = list(input_dims.keys())

        # 1) линейная проекция в общее пространство
        self.modality_proj = nn.ModuleDict({
            m: ModalityProjector(d, shared_in_dim)
            for m, d in input_dims.items()
        })

        # 2) единый эмоциональный энкодер
        self.shared_encoder = SharedEmotionEncoder(
            input_dim=shared_in_dim,
            hidden_dim=hidden_dim,
        )

        # 3) индивидуальные non-emotion энкодеры
        self.aux_encoders = nn.ModuleDict({
            m: ModalityAuxEncoder(d, hidden_dim)
            for m, d in input_dims.items()
        })

        # 4) одна emo-голова и PKL-головы по модальностям
        self.emo_head = EmotionClassifier(hidden_dim, emo_out_dim)
        self.pkl_heads = nn.ModuleDict({
            m: PersonalityRegressor(hidden_dim, pkl_out_dim)
            for m in self.modalities
        })

    # ─────────────────── forward ────────────────────
    def forward(self, x_dict: dict[str, torch.Tensor], modality: str):
        """
        x_dict : {modality: [B, D_m]}
        modality: ключ, который сейчас обрабатываем
        """
        x = x_dict[modality]                         # [B, D_m]

        # shared supra-modal эмо-вектор
        x_proj = self.modality_proj[modality](x)     # [B, shared_in_dim]
        z_emo  = self.shared_encoder(x_proj)         # [B, hidden_dim]

        # private не-эмо-вектор
        z_aux  = self.aux_encoders[modality](x)      # [B, hidden_dim]

        emo_logits = self.emo_head(z_emo)            # [B, 7]
        pkl_scores = self.pkl_heads[modality](z_aux) # [B, 5]

        return {
            "z_emo": z_emo,
            "z_aux": z_aux,
            "emotion_logits": emo_logits,
            "personality_scores": pkl_scores,
        }
