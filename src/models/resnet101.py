import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights

def get_model(num_classes=7):
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(2, 3, 224, 224)
    print("Output shape:", model(x).shape)