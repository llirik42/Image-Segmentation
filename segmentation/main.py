
from segmentation.segmentation.model import UNet

net = UNet()

total_params = sum(p.numel() for p in net.parameters())
print("Общее число параметров в модели:", total_params)
