"""
Implements a Facial Landmarks Dataset class for retrieving images 
and their corresponding keypoints data.
"""


import os

import matplotlib.image as mpimg
import numpy as np
from torch.utils.data import Dataset


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


class FacialLandmarksDataset(Dataset):
    """Face Landmarks Dataset."""

    def __init__(self, keypoints_file, images_dir, transforms=None):
        
        """Initializes a dataset of images and their corresponding keypoints.
        
        Arguments
        ---------
        :keypoints_file: Path to the csv file with keypoints for images.
        :images_dir: Path to folder containing images.
        :transforms: List of transformations to be applied on images.
        """
        
        # load keypoints from CSV file
        with open(keypoints_file, "r") as f:
            # load all lines from csv file with 
            # the exception of header line.
            csv_rows = f.readlines()[1:]
        
        # A function to convert a list of keypoint values to a numpy array
        cvt2np = lambda x: np.array(x).astype('float')
   
        # Map images to their keypoint data by
        # Obtaining a tuple of the form, (image_name, keypoints), for each image
        # NB: In the csv, the first item on each line is the image_name
        keypoints_data = [row.split(",") for row in csv_rows]
        self.keypoints_data = [
            (row[0], cvt2np(row[1:])) for row in keypoints_data
        ]
        
        self.images_dir = images_dir
        self.transforms = transforms

    def __len__(self):
        """Returns the count of samples in the dataset"""

        return len(self.keypoints_data)

    def __getitem__(self, index):
        """Retrieves an image and its corresponding keypoints using an index.
        
        Arguments
        ---------
        :index: Index of image to be retrieved
        """
        
        image_name, keypoints = self.keypoints_data[index]
        image_path = os.path.join(
            self.images_dir,
            image_name
        )
        image = mpimg.imread(image_path)
        
        # Convert images with more than 3 channels
        # to images with 3 channels
        if image.shape[2] > 3:
            image = image[:,:,0:3]
        
        # original keypoints list for each image contains 136 values.
        # They are concatenations of X and Y coordinates of 68 keypoints 
        # in the 2D image plane.
        
        # Below, the original 136-length keypoints list gets transformed into
        # 68 pairs of X, Y 
        # coordinates for each keypoint
        keypoints = keypoints.reshape(-1, 2)
        
        # Return image, keypoints pair as a tuple
        # And apply relevant transformations if applicable
        sample = (image, keypoints)
        if self.transforms:
            sample = self.transforms(sample)

        return sample
    
