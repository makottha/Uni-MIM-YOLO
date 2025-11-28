import torch
from copy import deepcopy


class ModelEMA:
    """
    Exponential Moving Average (EMA) for the Teacher Model.
    The Teacher is a slowly updating version of the Student.
    """

    def __init__(self, model, decay=0.999):
        # Create a deep copy of the model for the Teacher
        self.ema = deepcopy(model).eval()
        self.decay = decay

        # Detach parameters so gradients strictly flow to Student, never Teacher
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Formula: theta_t = decay * theta_t + (1 - decay) * theta_s
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(self.decay * v + (1 - self.decay) * msd[k])