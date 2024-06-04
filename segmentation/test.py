import torch
from torch import nn

class LesionSegmentation(nn.Module):
    def __init__(self):
        super(LesionSegmentation, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation='sigmoid'
        )

    def forward(self, x):
        # Perform forward pass through model
        mask = self.model(x)
        return mask