import os
import time
import torch
import argparse
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from segmentation.segmentation.model import UNet
from segmentation.dataset.massachusetts_roads import MassachusettsRoadsDataset
from config import config

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn(model,
          train_dataloader, test_dataloader,
          loss_function,
          optimizer,
          n_epoch=10,
          checkpoint_path='checkpoint.pth'):
    accuracy_on_train = []
    accuracy_on_test = []

    for epoch_n in range(1, n_epoch + 1):
        epoch_start_time = time.time()
        model.train()

        for (features, answers) in train_dataloader:
            features, answers = features.to(dev), answers.to(dev)
            model_predictions = model(features)

            loss = loss_function(model_predictions, answers)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            binary_model_tensor = torch.where(model_predictions > 0.5,
                                              torch.tensor(1, device=model_predictions.device),
                                              torch.tensor(0, device=model_predictions.device))
            accuracy_on_train.append(
                torch.sum(binary_model_tensor != answers).item() / (config['image_size'] * config['image_size'])
            )

        with torch.no_grad():
            model.eval()
            for (features, answers) in test_dataloader:
                features, answers = features.to(dev), answers.to(dev)

                model_predictions = model(features)
                binary_model_tensor = torch.where(model_predictions > 0.5,
                                                  torch.tensor(1, device=model_predictions.device),
                                                  torch.tensor(0, device=model_predictions.device))
                accuracy_on_test.append(
                    torch.sum(binary_model_tensor != answers).item() / (config['image_size'] * config['image_size'])
                )

        print("[INFO] EPOCH: {}/{}".format(epoch_n, n_epoch))
        print("Train accuracy: {:.6f}, Test accuracy: {:.4f}".format(
            accuracy_on_train[-1], accuracy_on_test[-1]))
        print("Epoch trained for: ", time.time() - epoch_start_time, " sec.")

    if checkpoint_path:
        print("Make checkpoint to :", checkpoint_path)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        torch.save(checkpoint, checkpoint_path)
    else:
        print("Learning done, don't make checkpoint")

    return accuracy_on_train, accuracy_on_test


train_transforms = v2.Compose([
    v2.CenterCrop(size=(256, 256))
    # v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomVerticalFlip(p=0.5)
])

test_transforms = v2.Compose([
    v2.CenterCrop(size=(256, 256))
])

mrd_train = MassachusettsRoadsDataset(train=True, transforms=train_transforms)
mrd_test = MassachusettsRoadsDataset(train=False, transforms=test_transforms)
train_dataloader = DataLoader(mrd_train, batch_size=16, shuffle=True, num_workers=os.cpu_count())
test_dataloader = DataLoader(mrd_test, batch_size=16, shuffle=True, num_workers=os.cpu_count())


def main():
    parser = argparse.ArgumentParser(description='UNet: binary road segmentation, learning module')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint if any', default=None)
    parser.add_argument('--epoch', type=int, help='Learning rate', default=50)
    parser.add_argument('-lr', type=float, help='Learning rate', default=0.1)
    args = parser.parse_args()

    net = UNet()
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    learn_start_time = time.time()

    accuracy_train, accuracy_test = learn(net,
                                          train_dataloader=train_dataloader,
                                          test_dataloader=test_dataloader,
                                          loss_function=loss_function,
                                          optimizer=optimizer,
                                          n_epoch=args.epoch,
                                          checkpoint_path=args.checkpoint
                                          )
    learn_end_time = time.time()

    print("Learning time: ", learn_end_time - learn_start_time)

    # print(accuracy_test)
    # print(accuracy_train)
    #
    # plt.plot(accuracy_train, label='train')
    # plt.plot(accuracy_test, label='test')
    #
    # plt.show()


if __name__ == '__main__':
    main()
