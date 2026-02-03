import torch.nn as nn
import torchvision.models as models


def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        print(model)
    elif model_name == 'efficientnet-b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        print(model)
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        print(model)
    elif model_name == 'vit_b16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        print(model)
    else:
        raise ValueError(f"Nieznany model: {model_name}")

    return model

if __name__ == "__main__":
    get_model('resnet50')
    # get_model('efficientnet-b0')
    # get_model('mobilenetv2')
    # get_model('vit_b16') //7 - freeze