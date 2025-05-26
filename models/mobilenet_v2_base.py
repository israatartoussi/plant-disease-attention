from torchvision.models import mobilenet_v2
import torch.nn as nn

def get_mobilenet_v2_base(num_classes):
    model = mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
