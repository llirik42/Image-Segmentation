import argparse
import random

import torch
from torchvision.transforms import v2
import matplotlib.pyplot as plt

from segmentation.segmentation.model import UNet
from segmentation.dataset.massachusetts_roads import MassachusettsRoadsDataset

from config import config

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = v2.Compose([
    v2.Resize(size=(int(config['image_size']), int(config['image_size']))),
    # v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
mrd_test = MassachusettsRoadsDataset(train=True, transforms=transforms)

def main():
    parser = argparse.ArgumentParser(description='UNet: binary road segmentation')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint', required=True)
    args = parser.parse_args()

    net = UNet()

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])

    n = random.randint(0, 20)
    s_img = mrd_test[n][0].unsqueeze(0)
    s_mask = mrd_test[n][1].unsqueeze(0)
    print(s_img)

    p_mask = net(s_img.to(dev))

    if s_img.is_cuda:
        s_img = s_img.cpu()
    if s_mask.is_cuda:
        s_mask = s_mask.cpu()
    if p_mask.is_cuda:
        p_mask = p_mask.cpu()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(s_img.squeeze().permute(2, 1, 0))
    plt.title('Source')

    plt.subplot(1, 3, 2)
    plt.imshow(s_mask.squeeze())
    plt.title('Source mask')

    plt.subplot(1, 3, 3)
    plt.imshow(p_mask.detach().squeeze().numpy())
    plt.title('Predicted mask')

    plt.show()


if __name__ == '__main__':
    main()
