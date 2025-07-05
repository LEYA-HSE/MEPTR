import torch
import torch.nn.functional as F
from transformers import CLIPProcessor
from utils.body.model_loader import get_fusion_model

class PretrainedImageEmbeddingExtractor:
    """
    Обёртка над предобученными fusion-моделями для body и face.
    Заодно хранит CLIP-процессор, чтобы им могли пользоваться другие модули.
    """

    def __init__(self, device="cuda",
                 clip_name: str = "openai/clip-vit-base-patch32"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.body_model = get_fusion_model("body", device)
        self.face_model = get_fusion_model("face", device)

    @torch.no_grad()
    def extract(self, *, body_tensor=None, face_tensor=None):
        """
        Возвращает dict с четырьмя ключами, как просила Е. Рюмина.
        Принимает батч тензоров формата [N, 3, H, W] — без лишних unsqueeze(0).
        """
        results = {}

        if body_tensor is not None:
            body_out = self.body_model(
                emotion_input=body_tensor.to(self.device),
                personality_input=body_tensor.to(self.device),
                return_features=True,
            )
            results["body"] = {
                "emotion_logits":           body_out["emotion_logits"].cpu(),
                "personality_scores":       body_out["personality_scores"].cpu(),
                "last_emo_encoder_features": body_out["last_emo_encoder_features"].cpu(),
                "last_per_encoder_features": body_out["last_per_encoder_features"].cpu(),
            }

        if face_tensor is not None:
            face_out = self.face_model(
                emotion_input=face_tensor.to(self.device),
                personality_input=face_tensor.to(self.device),
                return_features=True,
            )
            results["face"] = {
                "emotion_logits":           face_out["emotion_logits"].cpu(),
                "personality_scores":       face_out["personality_scores"].cpu(),
                "last_emo_encoder_features": face_out["last_emo_encoder_features"].cpu(),
                "last_per_encoder_features": face_out["last_per_encoder_features"].cpu(),
            }

        return results
