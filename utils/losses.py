import torch
import torch.nn as nn
import torch.nn.functional as F

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

def _binarize_with_nan(x: torch.Tensor, threshold=0.0):
    mask = ~torch.isnan(x)
    out = torch.zeros_like(x)
    out[mask] = (x[mask] > threshold).float()
    return out


class MultiTaskLossWithNaN_v2(nn.Module):
    """
    Двухзадачная версия (без AH/BAH):
      - эмоции: CE или BCEWithLogits (с поддержкой valid_emo и SSL по уверенностям)
      - персоналити: любой из loss_types (с поддержкой NaN по трейтам и SSL через BCE на уверенных значениях)
    Также реализован GradNorm-подобный апдейт отдельных весов для supervised/SSL компонент.
    """

    SUP_KEYS = ["emo_sup", "per_sup"]
    SSL_KEYS = ["emo_ssl", "per_ssl"]

    def __init__(
        self,
        # начальные «бюджеты» важности для supervised-веток
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,

        # class weights / опции лоссов
        emo_weights=None,
        personality_loss_type: str = "ccc",
        emotion_loss_type: str = "BCE",

        # гиперпараметры для некоторых лоссов персоналити
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,

        # SSL-пороги
        ssl_confidence_threshold_emo: float = 0.80,
        ssl_confidence_threshold_pt: float = 0.60,

        # GradNorm-гиперпараметры (раздельно для supervised и SSL «кошельков»)
        alpha_sup: float = 1.25,
        w_lr_sup: float = 0.025,
        alpha_ssl: float = 0.75,
        w_lr_ssl: float = 0.001,
        lambda_ssl: float = 0.2,
        w_floor: float = 1e-3,
    ):
        super().__init__()

        self.eps = float(eps)

        # --- конфигурация эмоций ---
        self.emotion_loss_type = emotion_loss_type
        if emotion_loss_type == "CE":
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
        elif emotion_loss_type == "BCE":
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
        else:
            raise ValueError(f"Unknown emotion_loss_type: {emotion_loss_type}")

        # --- конфигурация персоналити ---

        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(), "mse": MSELoss(),
            "bell": BellLoss(), "logcosh": LogCosh(), "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(), "rmse_bell": RMBell(), "rmse_logcosh": RMLCosh(), "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(), "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(), "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(), "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(f"Unknown personality_loss_type: {personality_loss_type}")
        self.personality_loss_type = personality_loss_type
        self.personality_loss = loss_types[personality_loss_type]

        # --- SSL-пороги ---
        self.ssl_confidence_threshold_emo = float(ssl_confidence_threshold_emo)
        self.ssl_confidence_threshold_pt = float(ssl_confidence_threshold_pt)

        # --- параметры GradNorm «кошельков» ---
        self.alpha_sup = float(alpha_sup)
        self.w_lr_sup = float(w_lr_sup)
        self.alpha_ssl = float(alpha_ssl)
        self.w_lr_ssl = float(w_lr_ssl)
        self.lambda_ssl = float(lambda_ssl)
        self.w_floor = float(w_floor)

        self.budget_sup = 2.0
        self.budget_ssl = 2.0 * self.lambda_ssl

        # обучаемые веса для supervised-компонент
        self.weight_sup = nn.ParameterDict({
            "emo_sup": nn.Parameter(torch.tensor(float(weight_emotion))),
            "per_sup": nn.Parameter(torch.tensor(float(weight_personality))),
        })

        # обучаемые веса для SSL-компонент (инициализируем одинаково = lambda_ssl)
        self.weight_ssl = nn.ParameterDict({
            "emo_ssl": nn.Parameter(torch.tensor(self.lambda_ssl, dtype=torch.float32)),
            "per_ssl": nn.Parameter(torch.tensor(self.lambda_ssl, dtype=torch.float32)),
        })

        self._normalize(self.weight_sup, self.SUP_KEYS, self.budget_sup)
        self._normalize(self.weight_ssl, self.SSL_KEYS, self.budget_ssl)

        # запоминаем стартовые значения лоссов для GradNorm-нормировки
        self.init_sup = {}
        self.init_ssl = {}

    # ---- helpers ----

    @staticmethod
    def _shared_params_from_model(model: nn.Module):
        # «Шареные» параметры — всё, что не относится к головам задач (emotion/personality)
        return [p for name, p in model.named_parameters()
                if ("emotion" not in name and "personality" not in name)]

    @staticmethod
    def _to_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(indices, num_classes=num_classes).float()

    @staticmethod
    def _mean_abs_norm(grads):
        g = [t.detach().flatten() for t in grads if t is not None]
        if not g:
            return None
        return torch.norm(torch.cat(g), p=2)

    @staticmethod
    def _safe_detach(x: torch.Tensor) -> torch.Tensor:
        return x.detach() if isinstance(x, torch.Tensor) else torch.tensor(float(x))

    def _normalize(self, pdict: nn.ParameterDict, keys, target_sum: float):
        with torch.no_grad():
            s = sum(pdict[k] for k in keys)
            sval = s.detach().clamp_min(1e-8)
            for k in keys:
                pdict[k].data = target_sum * (pdict[k].data / sval)

    # ---- сбор компонент лосса (supervised + SSL) ----
    def _collect_components(self, outputs, labels):
        comps = {}

        # --- Supervised: EMO ---
        pred_e = outputs.get("emotion_logits", None)
        emo_mask = labels.get("valid_emo", None)
        if pred_e is not None:
            if emo_mask is None:
                true_e = labels["emotion"]
                pred_e_sup = pred_e
                any_sup = True
            else:
                any_sup = emo_mask.any()
                if any_sup:
                    true_e = labels["emotion"][emo_mask]
                    pred_e_sup = pred_e[emo_mask]

            if pred_e is not None and (emo_mask is None or any_sup):
                if self.emotion_loss_type == "BCE":
                    true_e = _binarize_with_nan(true_e, threshold=0.0)
                    comps["emo_sup"] = self.emotion_loss(pred_e_sup, true_e)
                else:  # CE
                    target_e = (torch.argmax(true_e, dim=1) if true_e.dim() > 1 else true_e.long())
                    comps["emo_sup"] = self.emotion_loss(pred_e_sup, target_e)

        # --- SSL: EMO ---
        if pred_e is not None and emo_mask is not None:
            unlabeled = ~emo_mask
            if unlabeled.any():
                pred_u = pred_e[unlabeled]
                probs = torch.sigmoid(pred_u) if self.emotion_loss_type == "BCE" else torch.softmax(pred_u, dim=1)
                conf, pseudo = torch.max(probs, dim=1)
                c_mask = conf > self.ssl_confidence_threshold_emo
                if c_mask.any():
                    pred_c = pred_u[c_mask]
                    if self.emotion_loss_type == "BCE":
                        num_c = pred_c.size(1)
                        pseudo_c = self._to_onehot(pseudo[c_mask], num_c)
                        comps["emo_ssl"] = self.emotion_loss(pred_c, pseudo_c)
                    else:
                        comps["emo_ssl"] = self.emotion_loss(pred_c, pseudo[c_mask])

        # --- Supervised: PER ---
        pred_p = outputs.get("personality_scores", None)
        per_mask = labels.get("valid_per", None)
        if pred_p is not None:
            if per_mask is None:
                tp = labels["personality"]; pp = pred_p
                any_sup = True
            else:
                any_sup = per_mask.any()
                if any_sup:
                    tp = labels["personality"][per_mask]; pp = pred_p[per_mask]

            if pred_p is not None and (per_mask is None or any_sup):
                if self.personality_loss_type == "ccc":
                    loss_per = 0.0; valid_traits = 0
                    for i in range(tp.shape[1]):
                        tmask = ~torch.isnan(tp[:, i])
                        if tmask.any():
                            loss_per = loss_per + self.personality_loss(tp[tmask, i], pp[tmask, i])
                            valid_traits += 1
                    if valid_traits > 0:
                        comps["per_sup"] = loss_per / valid_traits
                else:
                    comps["per_sup"] = self.personality_loss(tp, pp)

        # --- SSL: PER ---
        if pred_p is not None and per_mask is not None:
            unlabeled = ~per_mask
            if unlabeled.any():
                pu = torch.clamp(pred_p[unlabeled], 0.0, 1.0)     # (U, 5)
                pseudo = (pu > 0.5).float()                        # (U, 5)
                c_mask = ((pu > self.ssl_confidence_threshold_pt) |
                          (pu < (1.0 - self.ssl_confidence_threshold_pt)))
                if c_mask.any():
                    bce_per_elem = F.binary_cross_entropy(pu, pseudo, reduction="none")
                    weighted = (bce_per_elem * c_mask.float()).sum()
                    tot = c_mask.sum().float()
                    if tot > 0:
                        comps["per_ssl"] = weighted / tot

        return comps

    # ---- GradNorm обновление «кошелька» весов ----
    def _gradnorm_update_wallet(self, comps, keys, init_dict, weight_pdict, alpha, w_lr, budget, shared_params):
        # зафиксировать начальные значения лоссов
        for k in keys:
            Li = comps.get(k, None)
            if (Li is not None) and (k not in init_dict) and torch.isfinite(Li).all():
                init_dict[k] = Li.detach().clamp_min(1e-8)

        active = [k for k in keys if (k in comps) and (k in init_dict)]
        if not active:
            return

        G_list, r_list, w_list = [], [], []
        for k in active:
            Li = comps[k]
            grads = torch.autograd.grad(Li, shared_params, retain_graph=True, allow_unused=True)
            grad_norm = self._mean_abs_norm(grads) or torch.tensor(0.0, device=Li.device)
            wk = weight_pdict[k]
            G_list.append(wk * grad_norm)
            r_list.append((Li.detach().clamp_min(1e-8) / init_dict[k]))
            w_list.append(wk)

        G_stack = torch.stack(G_list); r_stack = torch.stack(r_list)
        G_avg, r_avg = G_stack.mean(), r_stack.mean()

        gn_loss = 0.0
        for i, _ in enumerate(active):
            target = (G_avg * ((r_stack[i] / r_avg) ** alpha)).detach()
            gn_loss = gn_loss + torch.abs(G_stack[i] - target)

        grads_w = torch.autograd.grad(gn_loss, w_list, retain_graph=True, allow_unused=True)

        with torch.no_grad():
            for wk, gw in zip(w_list, grads_w):
                if gw is None:
                    continue
                wk.data -= w_lr * gw
                wk.data.clamp_(min=self.w_floor)

            self._normalize(weight_pdict, active, budget)

    # ---- основной проход ----
    def forward(self, outputs, labels, model=None, shared_params=None, return_details=True):
        if shared_params is None:
            if model is None:
                raise ValueError("Pass either `model` or `shared_params`.")
            shared_params = self._shared_params_from_model(model)

        comps = self._collect_components(outputs, labels)

        # если нечего считать — вернуть ноль на корректном девайсе
        if not comps:
            device = None
            for v in outputs.values():
                if isinstance(v, torch.Tensor):
                    device = v.device; break
            if device is None:
                device = torch.device("cpu")
            zero = torch.tensor(0.0, requires_grad=True, device=device)
            return (zero, {}) if return_details else zero

        # обновить «кошельки» весов для supervised и SSL
        self._gradnorm_update_wallet(comps, self.SUP_KEYS, self.init_sup, self.weight_sup,
                                     self.alpha_sup, self.w_lr_sup, self.budget_sup, shared_params)
        self._gradnorm_update_wallet(comps, self.SSL_KEYS, self.init_ssl, self.weight_ssl,
                                     self.alpha_ssl, self.w_lr_ssl, self.budget_ssl, shared_params)

        # суммарный лосс (веса детачим, как и в твоей v3)
        total = 0.0
        for k in self.SUP_KEYS:
            if k in comps:
                total = total + self.weight_sup[k].detach() * comps[k]
        for k in self.SSL_KEYS:
            if k in comps:
                total = total + self.weight_ssl[k].detach() * comps[k]

        if not return_details:
            return total

        details = {
            "components": {k: self._safe_detach(v).item() for k, v in comps.items()},
            "weights_sup": {k: self._safe_detach(self.weight_sup[k]).item() for k in self.SUP_KEYS},
            "weights_ssl": {k: self._safe_detach(self.weight_ssl[k]).item() for k in self.SSL_KEYS},
            "budget_sup": self.budget_sup,
            "budget_ssl": self.budget_ssl,
        }
        return total, details


class MultiTaskLossWithNaN_v2_FLAGGED(nn.Module):

    SUP_KEYS = ["emo_sup", "per_sup"]
    SSL_KEYS = ["emo_ssl", "per_ssl"]

    def __init__(
        self,
        # ---- базовые supervised веса (используются также как старт для GradNorm SUP) ----
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,

        # ---- типы лоссов ----
        emo_weights=None,
        personality_loss_type: str = "ccc",
        emotion_loss_type: str = "BCE",

        # ---- гиперпараметры для некоторых персоналити-лоссов ----
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,

        # ---- SSL параметры ----
        enable_emotion_ssl: bool = True,
        ssl_weight_emotion: float = 0.2,
        ssl_confidence_threshold_emo: float = 0.80,

        enable_personality_ssl: bool = True,
        ssl_weight_personality: float = 0.2,
        ssl_confidence_threshold_pt: float = 0.60,   # confident если >thr или <(1-thr)

        # ---- GradNorm «кошельки» (как в v2) ----
        enable_gradnorm: bool = True,   # True -> как v2; False -> статический микс
        alpha_sup: float = 1.25,
        w_lr_sup: float = 0.025,
        alpha_ssl: float = 0.75,
        w_lr_ssl: float = 0.001,
        lambda_ssl: float = 0.2,        # стартовые веса SSL-кошельков
        w_floor: float = 1e-3,

        # ---- Паритет форматов таргетов и логирование ----
        enforce_target_parity: bool = True,
        enable_details: bool = False,
    ):
        super().__init__()

        self.eps = float(eps)

        # ===== EMOTION =====
        self.emotion_loss_type = emotion_loss_type
        if emotion_loss_type == "CE":
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
        elif emotion_loss_type == "BCE":
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
        else:
            raise ValueError(f"Unknown emotion_loss_type: {emotion_loss_type}")

        # ===== PERSONALITY =====
        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(), "mse": MSELoss(),
            "bell": BellLoss(), "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(), "rmse_bell": RMBell(), "rmse_logcosh": RMLCosh(), "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(), "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(), "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(), "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(f"Unknown personality_loss_type: {personality_loss_type}")
        self.personality_loss_type = personality_loss_type
        self.personality_loss = loss_types[personality_loss_type]

        # ===== FLAGS / HYPERS =====
        self.enforce_target_parity = bool(enforce_target_parity)
        self.enable_details = bool(enable_details)

        self.enable_emotion_ssl = bool(enable_emotion_ssl)
        self.ssl_weight_emotion = float(ssl_weight_emotion)
        self.ssl_confidence_threshold_emo = float(ssl_confidence_threshold_emo)

        self.enable_personality_ssl = bool(enable_personality_ssl)
        self.ssl_weight_personality = float(ssl_weight_personality)
        self.ssl_confidence_threshold_pt = float(ssl_confidence_threshold_pt)

        self.enable_gradnorm = bool(enable_gradnorm)

        # статические веса (когда GradNorm выключен)
        self.static_weight_emo_sup = float(weight_emotion)
        self.static_weight_per_sup = float(weight_personality)

        # GradNorm настройки и инициализация (как в v2)
        self.alpha_sup = float(alpha_sup)
        self.w_lr_sup = float(w_lr_sup)
        self.alpha_ssl = float(alpha_ssl)
        self.w_lr_ssl = float(w_lr_ssl)
        self.lambda_ssl = float(lambda_ssl)
        self.w_floor = float(w_floor)

        if self.enable_gradnorm:
            # бюджеты как в v2
            self.budget_sup = 2.0
            self.budget_ssl = 2.0 * self.lambda_ssl

            # кошельки весов (обучаемые параметры)
            self.weight_sup = nn.ParameterDict({
                "emo_sup": nn.Parameter(torch.tensor(self.static_weight_emo_sup, dtype=torch.float32)),
                "per_sup": nn.Parameter(torch.tensor(self.static_weight_per_sup, dtype=torch.float32)),
            })
            self.weight_ssl = nn.ParameterDict({
                "emo_ssl": nn.Parameter(torch.tensor(self.lambda_ssl, dtype=torch.float32)),
                "per_ssl": nn.Parameter(torch.tensor(self.lambda_ssl, dtype=torch.float32)),
            })

            self._normalize(self.weight_sup, self.SUP_KEYS, self.budget_sup)
            self._normalize(self.weight_ssl, self.SSL_KEYS, self.budget_ssl)

            # стартовые значения лоссов Li(0) для нормировки
            self.init_sup = {}
            self.init_ssl = {}

    # ====================== helpers (оставлены внутри класса) ======================
    @staticmethod
    def _to_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(indices, num_classes=num_classes).float()

    @staticmethod
    def _binarize_with_nan(x: torch.Tensor, threshold=0.0):
        mask = ~torch.isnan(x)
        out = torch.zeros_like(x)
        out[mask] = (x[mask] > threshold).float()
        return out

    @staticmethod
    def _normalize(pdict: nn.ParameterDict, keys, target_sum: float):
        with torch.no_grad():
            s = sum(pdict[k] for k in keys)
            sval = s.detach().clamp_min(1e-8)
            for k in keys:
                pdict[k].data = target_sum * (pdict[k].data / sval)

    @staticmethod
    def _shared_params_from_model(model: nn.Module):
        # как в v2: «shared» это всё, что не принадлежит головам с подстроками "emotion" и "personality"
        return [p for name, p in model.named_parameters()
                if ("emotion" not in name and "personality" not in name)]

    @staticmethod
    def _mean_abs_norm(grads):
        vecs = [g.detach().flatten() for g in grads if g is not None]
        if not vecs:
            return None
        return torch.norm(torch.cat(vecs), p=2)

    @staticmethod
    def _safe_detach(x):
        return x.detach() if isinstance(x, torch.Tensor) else torch.tensor(float(x))

    @staticmethod
    def _device_from(outputs):
        for v in outputs.values():
            if isinstance(v, torch.Tensor):
                return v.device
        return torch.device("cpu")

    def _prepare_emotion_targets(self, y_emo):
        if not self.enforce_target_parity:
            return y_emo
        if self.emotion_loss_type == "BCE":
            # как в v2: бинаризация с NaN
            return self._binarize_with_nan(y_emo, threshold=0.0)
        else:  # CE
            return (y_emo.argmax(dim=1) if y_emo.dim() > 1 else y_emo.long())

    # ==================== сбор компонент (точно как в v2) ====================
    def _collect_components(self, outputs, labels):
        comps = {}

        # ----- Supervised: EMOTION -----
        pred_emotions = outputs.get("emotion_logits", None)
        mask_valid_emo = labels.get("valid_emo", None)
        if pred_emotions is not None:
            if mask_valid_emo is None:
                y_emo = labels["emotion"]; pred_emo_sup = pred_emotions; any_sup = True
            else:
                any_sup = mask_valid_emo.any()
                if any_sup:
                    y_emo = labels["emotion"][mask_valid_emo]
                    pred_emo_sup = pred_emotions[mask_valid_emo]

            if mask_valid_emo is None or any_sup:
                target_emo = self._prepare_emotion_targets(y_emo)
                comps["emo_sup"] = self.emotion_loss(pred_emo_sup, target_emo)

        # ----- SSL: EMOTION -----
        if (self.enable_emotion_ssl and self.ssl_weight_emotion > 0.0
                and pred_emotions is not None and mask_valid_emo is not None):
            unlabeled_mask = ~mask_valid_emo
            if unlabeled_mask.any():
                pred_emo_unl = pred_emotions[unlabeled_mask]
                if self.emotion_loss_type == "BCE":
                    probs = torch.sigmoid(pred_emo_unl)
                    conf, pseudo_idx = torch.max(probs, dim=1)
                    confident = conf > self.ssl_confidence_threshold_emo
                    if confident.any():
                        pred_conf = pred_emo_unl[confident]
                        num_classes = pred_conf.size(1)
                        pseudo_1h = self._to_onehot(pseudo_idx[confident], num_classes)
                        comps["emo_ssl"] = self.emotion_loss(pred_conf, pseudo_1h)
                else:
                    probs = torch.softmax(pred_emo_unl, dim=1)
                    conf, pseudo_idx = torch.max(probs, dim=1)
                    confident = conf > self.ssl_confidence_threshold_emo
                    if confident.any():
                        pred_conf = pred_emo_unl[confident]
                        comps["emo_ssl"] = self.emotion_loss(pred_conf, pseudo_idx[confident])

        # ----- Supervised: PERSONALITY -----
        pred_traits = outputs.get("personality_scores", None)
        mask_valid_per = labels.get("valid_per", None)
        if pred_traits is not None:
            if mask_valid_per is None:
                y_per = labels["personality"]; pred_per_sup = pred_traits; any_sup = True
            else:
                any_sup = mask_valid_per.any()
                if any_sup:
                    y_per = labels["personality"][mask_valid_per]
                    pred_per_sup = pred_traits[mask_valid_per]

            if mask_valid_per is None or any_sup:
                if self.personality_loss_type == "ccc":
                    loss_sum = 0.0; valid_dims = 0
                    for i in range(y_per.shape[1]):
                        dim_mask = ~torch.isnan(y_per[:, i])
                        if dim_mask.any():
                            loss_sum = loss_sum + self.personality_loss(y_per[dim_mask, i], pred_per_sup[dim_mask, i])
                            valid_dims += 1
                    if valid_dims > 0:
                        comps["per_sup"] = loss_sum / valid_dims   # как в v2: среднее по доступным трейтам
                else:
                    comps["per_sup"] = self.personality_loss(y_per, pred_per_sup)

        # ----- SSL: PERSONALITY (v2/v3 одинаковая идея порога) -----
        if (self.enable_personality_ssl and self.ssl_weight_personality > 0.0
                and pred_traits is not None and mask_valid_per is not None):
            unlabeled_mask = ~mask_valid_per
            if unlabeled_mask.any():
                preds_unl = torch.clamp(pred_traits[unlabeled_mask], 0.0, 1.0)  # важно для BCE
                pseudo = (preds_unl > 0.5).float()
                thr = self.ssl_confidence_threshold_pt
                confident = (preds_unl > thr) | (preds_unl < (1.0 - thr))
                tot = confident.sum().float()
                if tot > 0:
                    bce_per_elem = F.binary_cross_entropy(preds_unl, pseudo, reduction="none")
                    weighted = (bce_per_elem * confident.float()).sum()
                    comps["per_ssl"] = weighted / tot

        return comps

    # ==================== GradNorm обновление (как в v2) ====================
    def _gradnorm_update_wallet(self, comps, keys, init_dict, weight_pdict, alpha, w_lr, budget, shared_params):
        # зафиксировать стартовые значения Li(0)
        for key in keys:
            Li = comps.get(key, None)
            if (Li is not None) and (key not in init_dict) and torch.isfinite(Li).all():
                init_dict[key] = Li.detach().clamp_min(1e-8)

        active = [k for k in keys if (k in comps) and (k in init_dict)]
        if not active:
            return

        G_list, r_list, w_list = [], [], []
        for key in active:
            Li = comps[key]
            grads = torch.autograd.grad(Li, shared_params, retain_graph=True, allow_unused=True)
            gnorm = self._mean_abs_norm(grads) or torch.tensor(0.0, device=Li.device)
            wk = weight_pdict[key]
            G_list.append(wk * gnorm)
            r_list.append((Li.detach().clamp_min(1e-8) / init_dict[key]))
            w_list.append(wk)

        G = torch.stack(G_list); r = torch.stack(r_list)
        G_avg, r_avg = G.mean(), r.mean()

        gn_loss = 0.0
        for i, _ in enumerate(active):
            target = (G_avg * ((r[i] / r_avg) ** alpha)).detach()
            gn_loss = gn_loss + torch.abs(G[i] - target)

        grads_w = torch.autograd.grad(gn_loss, w_list, retain_graph=True, allow_unused=True)

        with torch.no_grad():
            for wk, gw in zip(w_list, grads_w):
                if gw is None:
                    continue
                wk.data -= w_lr * gw
                wk.data.clamp_(min=self.w_floor)

            self._normalize(weight_pdict, active, budget)

    # ============================== forward ===============================
    def forward(self, outputs, labels, model=None, shared_params=None, return_details=True):
        comps = self._collect_components(outputs, labels)

        if not comps:
            device = self._device_from(outputs)
            zero = torch.tensor(0.0, requires_grad=True, device=device)
            return (zero, {}) if (return_details and self.enable_details) else zero

        # --- режим как v2: GradNorm и «кошельки» ---
        if self.enable_gradnorm:
            if shared_params is None:
                if model is None:
                    raise ValueError("Pass either `model` or `shared_params` when enable_gradnorm=True.")
                shared_params = self._shared_params_from_model(model)

            # SUP и SSL кошельки (как v2)
            self._gradnorm_update_wallet(
                comps, self.SUP_KEYS, getattr(self, "init_sup", {}),
                self.weight_sup, self.alpha_sup, self.w_lr_sup,
                getattr(self, "budget_sup", 2.0), shared_params
            )
            self._gradnorm_update_wallet(
                comps, self.SSL_KEYS, getattr(self, "init_ssl", {}),
                self.weight_ssl, self.alpha_ssl, self.w_lr_ssl,
                getattr(self, "budget_ssl", 2.0 * self.lambda_ssl), shared_params
            )

            total = 0.0
            for k in self.SUP_KEYS:
                if k in comps:
                    total = total + self.weight_sup[k].detach() * comps[k]
            for k in self.SSL_KEYS:
                if k in comps:
                    total = total + self.weight_ssl[k].detach() * comps[k]

            if return_details and self.enable_details:
                details = {
                    "components": {k: float(self._safe_detach(v)) for k, v in comps.items()},
                    "weights_sup": {k: float(self._safe_detach(self.weight_sup[k])) for k in self.SUP_KEYS},
                    "weights_ssl": {k: float(self._safe_detach(self.weight_ssl[k])) for k in self.SSL_KEYS},
                }
                return total, details
            return total

        # --- статический режим (v3-стиль суммирования фикс. весами) ---
        total = 0.0
        if "emo_sup" in comps:
            total = total + self.static_weight_emo_sup * comps["emo_sup"]
        if "per_sup" in comps:
            total = total + self.static_weight_per_sup * comps["per_sup"]
        if "emo_ssl" in comps and self.enable_emotion_ssl and self.ssl_weight_emotion > 0.0:
            total = total + self.ssl_weight_emotion * comps["emo_ssl"]
        if "per_ssl" in comps and self.enable_personality_ssl and self.ssl_weight_personality > 0.0:
            total = total + self.ssl_weight_personality * comps["per_ssl"]

        if return_details and self.enable_details:
            details = {
                "components": {k: float(self._safe_detach(v)) for k, v in comps.items()},
                "weights_static": {
                    "emo_sup": float(self.static_weight_emo_sup),
                    "per_sup": float(self.static_weight_per_sup),
                    "emo_ssl": float(self.ssl_weight_emotion),
                    "per_ssl": float(self.ssl_weight_personality),
                },
            }
            return total, details

        return total
