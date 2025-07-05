# coding: utf-8
import torch, toml
from pathlib import Path
from models.models import EmotionMamba, PersonalityMamba, FusionTransformer


# ------------------------------------------------------------------
# Таблица «модальность → TOML + чек-пойнт»
# ------------------------------------------------------------------
MODALITY_META = {
    "body": {
        "toml": Path("inference_config_body.toml"),
        "ckpt": "extractors/body/clip_body_mamba_transformer_fusion_model.pt",
    },
    "face": {
        "toml": Path("inference_config_face.toml"),
        "ckpt": "extractors/face/clip_face_mamba_transformer_fusion_model.pt",
    },
}


def _parse_toml(toml_path: Path) -> dict:
    """Берём только числовые гиперпараметры; путь к ckpt игнорируем."""
    d = toml.load(toml_path)["train"]["model"]
    emb = toml.load(toml_path)["embeddings"]
    return {
        "embedding_dim":   emb["image_embedding_dim"],
        "len_seq":         emb["counter_need_frames"],

        "hidden_dim_emo":       d["hidden_dim_emo"],
        "out_features_emo":     d["out_features_emo"],
        "tr_layer_number_emo":  d["tr_layer_number_emo"],
        "num_heads_emo":        d["num_transformer_heads_emo"],
        "pos_enc_emo":          d["positional_encoding_emo"],
        "mamba_d_state_emo":    d["mamba_d_state_emo"],
        "mamba_layers_emo":     d["mamba_layer_number_emo"],

        "hidden_dim_per":       d["hidden_dim_per"],
        "out_features_per":     d["out_features_per"],
        "tr_layer_number_per":  d["tr_layer_number_per"],
        "num_heads_per":        d["num_transformer_heads_per"],
        "pos_enc_per":          d["positional_encoding_per"],
        "mamba_d_state_per":    d["mamba_d_state_per"],
        "mamba_layers_per":     d["mamba_layer_number_per"],

        "per_activation":       d.get("best_per_activation", "sigmoid"),
        "dropout":              d["dropout"],
    }


def _make_branch(cls, cfg, *, is_emotion, device):
    p = "emo" if is_emotion else "per"
    return cls(
        input_dim_emotion     = cfg["embedding_dim"],
        input_dim_personality = cfg["embedding_dim"],
        len_seq               = cfg["len_seq"],
        hidden_dim            = cfg[f"hidden_dim_{p}"],
        out_features          = cfg[f"out_features_{p}"],
        per_activation        = cfg["per_activation"],
        tr_layer_number       = cfg[f"tr_layer_number_{p}"],
        num_transformer_heads = cfg[f"num_heads_{p}"],
        positional_encoding   = cfg[f"pos_enc_{p}"],
        mamba_d_model         = cfg[f"mamba_d_state_{p}"],
        mamba_layer_number    = cfg[f"mamba_layers_{p}"],
        dropout               = cfg["dropout"],
        num_emotions          = 7,
        num_traits            = 5,
        device                = device,
    ).to(device).eval()


def get_fusion_model(modality: str, device: str = "cuda") -> FusionTransformer:
    if modality not in MODALITY_META:
        raise ValueError("modality must be 'body' or 'face'")

    meta = MODALITY_META[modality]
    cfg  = _parse_toml(meta["toml"])

    emo_model = _make_branch(EmotionMamba,    cfg, is_emotion=True,  device=device)
    per_model = _make_branch(PersonalityMamba, cfg, is_emotion=False, device=device)

    model = FusionTransformer(
        emo_model             = emo_model,
        per_model             = per_model,
        input_dim_emotion     = cfg["embedding_dim"],
        input_dim_personality = cfg["embedding_dim"],
        hidden_dim            = cfg["hidden_dim_emo"],
        out_features          = cfg["out_features_emo"],
        per_activation        = cfg["per_activation"],
        tr_layer_number       = cfg["tr_layer_number_emo"],
        num_transformer_heads = cfg["num_heads_emo"],
        positional_encoding   = cfg["pos_enc_emo"],
        mamba_d_model         = cfg["mamba_d_state_emo"],
        mamba_layer_number    = cfg["mamba_layers_emo"],
        dropout               = cfg["dropout"],
        num_emotions          = 7,
        num_traits            = 5,
        device                = device,
    ).to(device).eval()

    # ←––– загружаем жёстко заданный checkpoint –––→
    state = torch.load(meta["ckpt"], map_location=device)
    model.load_state_dict(state)        # strict=True

    return model
