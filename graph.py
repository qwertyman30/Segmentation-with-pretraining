import random
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import torchvision.transforms.functional as TF
from torchvision.transforms import *
from data.segmentation import statistics
from data.pretraining import DataReaderPlainImg, custom_collate
from data.transforms import get_transforms_pretraining
from models.pretraining_backbone import ResNet18Backbone

class ImgRotation:
    """ Produce 4 rotated versions of the input image. """
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        """
        Produce 4 rotated versions of the input image.
        Args:
            img: the input PIL image to be rotated.
        Returns:
            rotated_imgs: a list containing all the rotated versions of img.
            labels: a list containing the corresponding label for each rotated image in rotated_imgs.
        """
        rotated_imgs = [TF.rotate(img, i*90) for i in range(4)]
        labels = range(4)
        assert len(rotated_imgs) == len(labels)
        return rotated_imgs, labels


class ApplyAfterRotations:
    """ Apply a transformation to a list of images (e.g. after applying ImgRotation)"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        images, labels = x
        images = [self.transform(i) for i in images]
        return images, labels


class ToTensorAfterRotations:
    """ Transform a list of images to a pytorch tensor (e.g. after applying ImgRotation)"""
    def __call__(self, x):
        images, labels = x
        return [TF.to_tensor(i) for i in images], [torch.tensor(l) for l in labels]


def get_transforms_pretraining():
    """ Returns the transformations for the pretraining task. """
    size = [256]*2
    train_transform = Compose([
        Resize(size),
        RandomCrop(size, pad_if_needed=True),
        ImgRotation(),
        ApplyAfterRotations(RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)),
        ToTensorAfterRotations(),
        ApplyAfterRotations(Normalize(statistics['mean'], statistics['std']))
    ])
    val_transform = Compose([Resize(size), RandomCrop(size, pad_if_needed=True),
                             ImgRotation(), ToTensorAfterRotations(),
                             ApplyAfterRotations(Normalize(statistics['mean'], statistics['std']))])
    return train_transform, val_transform


model = ResNet18Backbone(pretrained=False)
criterion = nn.CrossEntropyLoss()

data_root = "/home/mbengt/workspace/dl_lab/crops"
train_transform, val_transform = get_transforms_pretraining()
train_data = DataReaderPlainImg(os.path.join(data_root, str(256), "train"), transform=train_transform)
val_data = DataReaderPlainImg(os.path.join(data_root, str(256), "val"), transform=val_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2,
                                           pin_memory = True, drop_last=True, collate_fn=custom_collate)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2,
                                         pin_memory = True, drop_last=True, collate_fn=custom_collate)

train_losses, val_losses, val_accs = [], [], []
for i in range(10):
    with torch.no_grad():
        losses = []
        model.load_state_dict(torch.load(f'epoch_{i}.pth', map_location=torch.device('cpu')))
        for X_train, y_train in train_loader:
            y_preds = model(X_train)
            loss = criterion(y_preds, y_train)
            _, y_preds = y_preds.max(1)
            losses.append(loss.item())
        print('train', loss.item())
        train_losses.append(np.mean(losses))
        
        for X_val, y_val in val_loader:
            y_preds = model(X_val)
            loss = criterion(y_preds, y_val)
            _, y_preds = y_preds.max(1)
            accuracy = 100.0 * sum(y_val == y_preds)/len(y_preds)
            losses.append(loss.item())
            accuracies.append(accuracy.item())
        print('val', loss.item(), accuracy.item())
    val_losses.append(np.mean(losses))
    val_accs.append(np.mean(accuracies))
    
_, axes = plt.subplots(1, 2, figsize=(20, 10))
axes[0].plot(range(1, 11), losses)
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].set_title("Train")

axes[1].plot(range(1, 11), val_losses)
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Loss")
axes[1].set_title("Validation losses")

axes[2].plot(range(1, 11), val_accs)
axes[2].set_xlabel("Epochs")
axes[2].set_ylabel("Accuracy")
axes[2].set_title("Validation accuracies")
plt.savefig("Results.png")