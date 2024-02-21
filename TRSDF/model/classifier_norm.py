import torch
import torch.nn as nn
import torch.nn.functional as F
from .circlesoftmax import CircleSoftmax


class Classifier_norm(nn.Module):  # trick 1
    def __init__(self, planes, num_classes):
        super(Classifier_norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, planes))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        logits = F.linear(F.normalize(x), F.normalize(self.weight)) # x[b, 768]  w[num, 768]
        return logits





class Classifier_norm_circle(nn.Module):   # trick1+2，带有circlesoftmax的classifier_norm层
    def __init__(self, planes, num_classes):
        super(Classifier_norm_circle, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, planes))
        nn.init.normal_(self.weight, std=0.01)
        self.circle = CircleSoftmax(num_classes=num_classes, scale=64, margin=0.35)

    def forward(self, x):
        logits = F.linear(F.normalize(x), F.normalize(self.weight))
        logits = self.circle(logits)
        return logits