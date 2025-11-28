import torch
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average (EMA) for the Teacher Model.
    The Teacher is a slowly updating version of the Student.
    """

    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model).eval()  # Teacher is always in Eval mode
        self.decay = decay

        # Detach parameters to prevent gradient flow
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # theta_t = alpha * theta_t + (1 - alpha) * theta_s
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(self.decay * v + (1 - self.decay) * msd[k])