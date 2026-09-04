import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def get_model(num_classes=7):
    weights = EfficientNet_V2_S_Weights.DEFAULT

    model = efficientnet_v2_s(weights=weights)

    # Replace final classification layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_model()

    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    print("Output shape:", output.shape)