"""
Implements loaders for training and testing data.
"""

from torch.utils.data import DataLoader
from torchvision import transforms, utils

from dataset import FacialLandmarksDataset
from transformations import (
    Rescale,
    RandomCrop, 
    Normalize, 
    ToTensor
)


data_transform = transforms.Compose([
    Rescale(250),
    RandomCrop(224), 
    Normalize(), 
    ToTensor()
])

train_dataset = FacialLandmarksDataset(
    keypoints_file='/data/training_frames_keypoints.csv',
    images_dir='/data/training/',
    transforms=data_transform
)

test_dataset = FacialLandmarksDataset(
    keypoints_file='/data/test_frames_keypoints.csv',
    images_dir='/data/test/',
    transforms=data_transform
)

def dataloaders(batch_size=10):
    """Returns loaders for training and testing data."""

    trainloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )

    testloader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    return trainloader, testloader
