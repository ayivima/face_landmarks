"""
Implements loaders for training and testing data.
"""

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import FacialLandmarksDataset
from transformations import (
    Rescale,
    RandomCrop, 
    Normalize, 
    ToTensor
)


__author__ = "Victor mawusi Ayi <ayivima@hotmail.com>"


data_transform = Compose([
    Rescale(250),
    RandomCrop(224), 
    Normalize(), 
    ToTensor()
])

def train_dataset(transforms_pipe=data_transform):
    return FacialLandmarksDataset(
        keypoints_file='/data/training_frames_keypoints.csv',
        images_dir='/data/training/',
        transforms=transforms_pipe
    )

def test_dataset(transforms_pipe=data_transform):
    return FacialLandmarksDataset(
        keypoints_file='/data/test_frames_keypoints.csv',
        images_dir='/data/test/',
        transforms=transforms_pipe
    )

def trainloader(batch_size=10): 
    return DataLoader(
        train_dataset(), 
        batch_size=batch_size,
        shuffle=True
    )

def testloader(batch_size=10): 
    return DataLoader(
        test_dataset(), 
        batch_size=batch_size,
        shuffle=True
    )

def dataloaders(batch_size=10):
    """Returns loaders for training and testing data."""
    
    return (
        trainloader(batch_size),
        testloader(batch_size)
    )

