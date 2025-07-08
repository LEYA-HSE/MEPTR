import torch
from transformers import CLIPProcessor,CLIPModel
from modalities.video.model_loader import get_fusion_model

class PretrainedImageEmbeddingExtractor:
    """
    Обёртка над предобученными fusion-моделями для body, face, scene.
    Заодно хранит CLIP-процессор, чтобы им могли пользоваться другие модули.
    """

    def __init__(self, device="cuda",
                 clip_name: str = "openai/clip-vit-base-patch32"):
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        self.body_model = get_fusion_model("body", device)
        self.face_model = get_fusion_model("face", device)
        self.scene_model = get_fusion_model("scene", device)

    @torch.no_grad()
    def extract(self, *, body_tensor=None, face_tensor=None, scene_tensor=None):
        """
        Возвращает dict с четырьмя ключами.
        Принимает батч тензоров формата [N, 3, H, W] — без лишних unsqueeze(0).
        """
        results = {}

        if body_tensor is not None:
            body_tensor = self.clip_model.get_image_features(body_tensor.to(self.device))
            body_out = self.body_model(
                emotion_input       = torch.unsqueeze(body_tensor, 0),
                personality_input   = torch.unsqueeze(body_tensor, 0),
                return_features     = True,
            )
            results["body"] = {
                "emotion_logits": body_out["emotion_logits"].cpu(),
                "personality_scores": body_out["personality_scores"].cpu(),
                "last_emo_encoder_features": body_out["last_emo_encoder_features"].cpu(),
                "last_per_encoder_features": body_out["last_per_encoder_features"].cpu(),
            }

        if face_tensor is not None:
            face_tensor = self.clip_model.get_image_features(face_tensor.to(self.device))
            face_out = self.face_model(
                emotion_input=torch.unsqueeze(face_tensor, 0),
                personality_input=torch.unsqueeze(face_tensor, 0),
                return_features=True,
            )
            results["face"] = {
                "emotion_logits":           face_out["emotion_logits"].cpu(),
                "personality_scores":       face_out["personality_scores"].cpu(),
                "last_emo_encoder_features": face_out["last_emo_encoder_features"].cpu(),
                "last_per_encoder_features": face_out["last_per_encoder_features"].cpu(),
            }

        if scene_tensor is not None:
            scene_tensor = self.clip_model.get_image_features(scene_tensor.to(self.device))
            scene_out = self.scene_model(
                emotion_input     = torch.unsqueeze(scene_tensor, 0),
                personality_input = torch.unsqueeze(scene_tensor, 0),
                return_features   = True,
            )
            results["scene"] = {
                "emotion_logits": scene_out["emotion_logits"].cpu(),
                "personality_scores": scene_out["personality_scores"].cpu(),
                "last_emo_encoder_features": scene_out["last_emo_encoder_features"].cpu(),
                "last_per_encoder_features": scene_out["last_per_encoder_features"].cpu(),
            }

        return results
