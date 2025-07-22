import torch
import torch.nn as nn


class BellLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_p = torch.pow((y - p), 2)
        y_p_div = -1.0 * torch.div(y_p, 162.0)
        exp_y_p = torch.exp(y_p_div)
        loss = 300 * (1.0 - exp_y_p)
        return torch.mean(loss)


class LogCosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(p - y))
        return torch.mean(loss)


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(p, y))


class GL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.sigma = sigma

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gl = self.eps / (self.lam ** 2) * (1 - torch.exp(-1 * ((y - p) ** 2) / (self.sigma ** 2)))
        return gl.mean()


class RMBell(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y)


class RMLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.logcosh(p, y)


class RMGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.gl(p, y)


class RMBellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.logcosh(p, y)


class RMBellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.gl(p, y)


class BellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y)


class BellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.gl(p, y)


class BellLCoshGL(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()
        self.gl = GL()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y) + self.gl(p, y)


class LogCoshGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.logcosh = LogCosh()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logcosh(p, y) + self.gl(p, y)

class MAELoss(nn.Module):
    """Mean Absolute Error loss"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x - y))

class MSELoss(nn.Module):
    """Mean Squared Error loss"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.pow(x - y, 2))

class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    Measures the agreement between two variables

    It is a product of
    - precision (pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation
    - rho =  1: perfect agreement
    - rho =  0: no agreement
    - rho = -1: perfect disagreement

    Args:
        eps (float, optional): Avoiding division by zero. Defaults to 1e-8.

    original: https://github.com/DresvyanskiyDenis/ABAW_2024/blob/main/src/audio/loss/loss.py
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes CCC loss

        Args:
            x (torch.Tensor): Input tensor
            y (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: 1 - CCC loss value
        """
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc

class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weight_emotion=1.0,
        weight_personality=1.0,
        emo_weights=None,
        personality_loss_type="ccc",  # см. ниже список типов
        eps=1e-8,
        lam_gl=1.0,
        eps_gl=600,
        sigma_gl=8
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality

        # Эмоции — всегда CrossEntropy
        self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)

        # Персональные качества — выбираем по имени
        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }

        if personality_loss_type not in loss_types:
            raise ValueError(f"Unknown personality_loss_type: {personality_loss_type}. "
                             f"Available: {list(loss_types.keys())}")

        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0

        # Эмоции (классификация)
        if 'emotion_logits' in outputs and 'emotion' in labels:
            true_emotion = labels['emotion']
            pred_emotion = outputs['emotion_logits']
            loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # Персональные качества (регрессия)
        if 'personality_scores' in outputs and 'personality' in labels:
            true_personality = labels['personality']
            pred_personality = outputs['personality_scores']

            if self.personality_loss_type == "ccc":
                loss_per = 0.0
                for i in range(5):  # по каждому из 5 признаков
                    loss_per += self.personality_loss(true_personality[:, i], pred_personality[:, i])
                loss += (loss_per) * self.weight_personality
                # loss += (loss_per / 5.0) * self.weight_personality
            else:
                loss += self.weight_personality * self.personality_loss(true_personality, pred_personality)

        return loss

def binarize_with_nan(x, threshold=0.5):
    # Создаем маску NaN
    nan_mask = torch.isnan(x)

    # Бинаризуем (не затрагивая NaN)
    binary = torch.zeros_like(x)
    binary[x > threshold] = 1.0

    # Восстанавливаем NaN там, где они были
    binary[nan_mask] = float('nan')

    return binary


class MultiTaskLossWithNaN(nn.Module):
    def __init__(
        self,
        weight_emotion=1.0,
        weight_personality=1.0,
        emo_weights=None,
        personality_loss_type="ccc",  # см. ниже список типов
        emotion_loss_type='BCE',
        eps=1e-8,
        lam_gl=1.0,
        eps_gl=600,
        sigma_gl=8
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality

        # Эмоции — всегда CrossEntropy
        if emotion_loss_type == 'CE':
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        if emotion_loss_type == 'BCE':
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type

        # Персональные качества — выбираем по имени
        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }

        if personality_loss_type not in loss_types:
            raise ValueError(f"Unknown personality_loss_type: {personality_loss_type}. "
                             f"Available: {list(loss_types.keys())}")

        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0

        # Emotion branch
        emo_mask = labels['valid_emo']
        pred_emotion = outputs.get('emotion_logits')
        if pred_emotion is not None and emo_mask.any():
            true_emotion = labels['emotion'][emo_mask]
            pred_emotion = pred_emotion[emo_mask]
            # pred_emotion = outputs['emotion_logits'][emo_mask]

            if self.emotion_loss_type == 'BCE':
                true_emotion = binarize_with_nan(true_emotion, threshold=0)

            loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # Personality branch
        per_mask = labels['valid_per']
        pred_personality = outputs.get('personality_scores')
        if pred_personality is not None and per_mask.any():
            true_personality = labels['personality'][per_mask]
            # pred_personality = outputs['personality_scores'][per_mask]
            pred_personality = pred_personality[per_mask]

            if self.personality_loss_type == "ccc":
                loss_per = 0.0
                valid_traits = 0
                for i in range(5):
                    trait_mask = ~torch.isnan(true_personality[:, i])
                    if trait_mask.any():
                        loss_per += self.personality_loss(
                            true_personality[trait_mask, i],
                            pred_personality[trait_mask, i]
                        )
                        valid_traits += 1

                if valid_traits > 0:
                    loss += (loss_per / valid_traits) * self.weight_personality
            else:
                loss += self.weight_personality * self.personality_loss(
                    true_personality,
                    pred_personality
                )
        if not isinstance(loss, torch.Tensor):
            device = (
                outputs.get("emotion_logits", None).device
                if outputs.get("emotion_logits", None) is not None
                else outputs.get("personality_scores", torch.tensor(0.0)).device
            )
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return loss
