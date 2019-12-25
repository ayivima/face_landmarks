"""
Implements classes for transformations and augmentation of 
images and their corresponding keypoints.
"""


import torch
from numpy import random
from cv2 import (
    COLOR_RGB2GRAY,
    cvtColor,
    resize
)


class Normalize(object):
    """Normalizes image pixel intensities to values between 0 and 1, 
    and keypoints to between -1 and 1"""        
    
    def __init__(self, keypoints_mean=100, keypoints_std=50):
        """Initializes mean and standard deviation of keypoints.
        
        Arguments
        ---------
        :keypoints_mean: Mean of keypoint values
        :keypoints_std: Standard Deviation of keypoint values 
        """
        
        self.keypoints_mean = keypoints_mean
        self.keypoints_std = keypoints_std
    
    def __call__(self, sample):
        
        image, keypoints = sample
        image_gray = cvtColor(image, COLOR_RGB2GRAY)
        
        # Z-score normalization of keypoints
        # Normalization of keypoints to 0 to 1
        keypoints_norm = (keypoints - self.keypoints_mean) / self.keypoints_std
        image_norm = image_gray / 255.0

        return image_norm, keypoints_norm


class Rescale(object):
    """Rescale the image in a sample to a given size"""

    def __init__(self, output_size):
        """Initializes the size of the image after rescaling.
        
        Arguments
        ---------
        output_size (tuple or int): Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
        """
        
        if type(output_size) in (int, tuple):
            self.output_size = output_size
        else:
            raise TypeError(
                "Output size must be of type 'tuple' or 'int'"
            )

    def __call__(self, sample):
        """Rescales an image and its corresponding keypoints to a specified size.
        
        Arguments
        ---------
        :sample: a tuple containing an image and its keypoints.

        """
        
        image, keypoints = sample
        height, width = image.shape[:2]
        
        # function to resize image while maintaining aspect ratio
        new_sizes = lambda h, w, n: (
            ((n * h / w), n) if h > w else (n, (n * w / h))
        )
        
        if isinstance(self.output_size, int):
            new_height, new_width = [
                int(i) for i in new_sizes(height, width, self.output_size)
            ]
        else:
            new_height, new_width = [int(i) for i in self.output_size]

        # resize image using cv2.resize
        # scale keypoints accordingly
        img = resize(image, (new_width, new_height))
        keypoints = keypoints * [new_width / width, new_height / height]

        return (img, keypoints)


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_size):
        """Initializes crop dimensions."""
        
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        elif isinstance(crop_size, tuple):
            self.crop_size = crop_size
        else:
            raise TypeError(
                "Output Size argument must be of type 'int' or 'tuple'"
            )

    def __call__(self, sample):
        """Randomly crops a part of an image, to a specified size.
        
        Arguments
        ---------
        :sample: a tuple containing an image and its keypoints.

        """

        image, keypoints = sample
        height, width = image.shape[:2]
        new_height, new_width = self.crop_size

        top = random.randint(0, height - new_height)
        left = random.randint(0, width - new_width)

        image = image[
            top: top + new_height,
            left: left + new_width
        ]

        keypoints = keypoints - [left, top]

        return (image, keypoints)


class ToTensor(object):
    """Converts images and their corresponding keypoints to Tensors."""

    def __call__(self, sample):
        """Converts an image and its corresponding keypoints to Tensors.
        
        Arguments
        ---------
        :sample: a tuple containing an image and its keypoints.

        """
        
        image, keypoints = sample
         
        # if image has no grayscale color channel,
        # add the channel dimension
        imshape = image.shape
        if(len(imshape) == 2):
            H, W = imshape
            image = image.reshape(H, W, 1)
            
        # Adapt image dimensions for pytorch image formatting
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return (
            torch.from_numpy(image),
            torch.from_numpy(keypoints)
        )
