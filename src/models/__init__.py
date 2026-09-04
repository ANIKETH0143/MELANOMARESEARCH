from .efficientnetv2 import get_model as efficientnetv2
from .densenet121 import get_model as densenet121
from .resnet101 import get_model as resnet101
from .convnext import get_model as convnext
from .swin import get_model as swin


def get_model(model_name, num_classes=7):

    if model_name == "efficientnetv2":
        return efficientnetv2(num_classes)

    elif model_name == "densenet121":
        return densenet121(num_classes)

    elif model_name == "resnet101":
        return resnet101(num_classes)

    elif model_name == "convnext":
        return convnext(num_classes)

    elif model_name == "swin":
        return swin(num_classes)

    else:
        raise ValueError(
            "Unknown model. Choose: "
            "efficientnetv2, densenet121, resnet101, convnext, swin"
        )