import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def get_model(num_classes=7):
    weights = DenseNet121_Weights.DEFAULT

    model = densenet121(weights=weights)

    # Replace final classifier
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_model()

    x = torch.randn(2, 3, 224, 224)
    output = model(x)

    print("Output shape:", output.shape)