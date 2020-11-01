import torchvision.transforms as transforms
import numpy as np
import torch


def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
    y = (x - mean) / std_adj
    return y
#################################### full transform
train_full_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     transforms.ColorJitter(brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.3
                            ),
     # transforms.CenterCrop((64,64)),
     # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     transforms.RandomPerspective(distortion_scale=0.2,p=0.3),
     transforms.RandomResizedCrop(size=(64,64), scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=2),

     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.Lambda(lambda crops: torch.unsqueeze(crops, 0)),
     # transforms.RandomErasing()
     ])


test_full_transform = transforms.Compose(
    [
        transforms.Resize((64,64)),
        # transforms.Resize((73,73)),
        # transforms.CenterCrop((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Lambda(lambda crops: torch.unsqueeze(crops, 0)),

    ])



crop_full_transform = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.FiveCrop((45,45)),
        transforms.Lambda(lambda crops: torch.stack(
               [mid_transform(crop) for crop in
                crops])),

    ])

mid_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])





#################################### kinFaceW transform
train_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     transforms.ColorJitter(brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.3
                            ),
     # transforms.CenterCrop((64,64)),
     # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.4),
     transforms.RandomPerspective(distortion_scale=0.1,p=0.3),
     transforms.RandomResizedCrop(size=(64,64), scale=(0.9, 1.05), ratio=(0.97, 1.05), interpolation=2),
     # transforms.RandomResizedCrop(size=(64, 64), scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=2),

        # transforms.RandomErasing(),
     # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.RandomErasing()
     ])


test_transform = transforms.Compose(
    [
     transforms.Resize((64,64)),
     # transforms.Resize((73,73)),
     # transforms.CenterCrop((64, 64)),

        # transforms.Grayscale(num_output_channels=3),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


######################################### faceNet transform

facenet_testrans = transforms.Compose(
     [
     transforms.Resize((160,160)),
     np.float32,
     transforms.ToTensor(),
     prewhiten
     # transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

facenet_trans = transforms.Compose(
     [
     transforms.Resize((160,160)),
     transforms.ColorJitter(brightness=0.3,
                            contrast=0.3,
                            saturation=0.3,
                            hue=0.3
                            ),
     # transforms.RandomAffine(degrees=1,scale=(0.98,1.02),shear =1),
     transforms.RandomGrayscale(),
     transforms.RandomHorizontalFlip(p=0.2),
     transforms.RandomPerspective(distortion_scale=0.05,p=0.2),
     transforms.RandomResizedCrop(size=(160,160), scale=(0.98, 1.02), ratio=(0.98, 1.02), interpolation=2),
     np.float32,
     transforms.ToTensor(),
     # transforms.RandomErasing(),
     prewhiten
     # transforms.ToTensor(),
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])











