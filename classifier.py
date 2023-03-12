import numpy as np
import torch
from torch import nn


class AudioANNClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.ann = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=out_features),
        )

    def forward(self, x):
        x = self.ann(x)
        return x


@torch.no_grad()
def infer(classifier: nn.Module, data, mean: np.ndarray, std: np.ndarray, device='cpu'):
    classifier.eval()
    classifier = classifier.to(device)

    data = (data - mean) / std
    outs = classifier(data)

    _, pred_id = torch.max(outs, 1)
    probs = torch.softmax(outs, dim=1)[0].detach()
    return probs[pred_id].item(), int(pred_id.detach())
