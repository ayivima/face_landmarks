"""
Implements functions for displaying keypoints on images.
"""


import torch
import matplotlib.pyplot as plt
import numpy as np


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


# Set up for GPU use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fltTensor = (
    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
)
        
def plotter(
    model, 
    test_loader, 
    plotrows=5, 
    plotcolumns=8, 
    showactual=False,
    figsize=(17,10),
    markersize=20
):
    """Displays images with an overlay of predicted keypoints, 
    and (optionally) actual kepoints.
    
    Arguments
    ---------
    :model: A trained (or untrained) model.
    :test_loader: A generator for loading image and keypoint data.
    :plotrows: The number of rows for image plotting.
    :plotcolumns: The number of columns for image plotting. 
                  Value must not exceed the batch size of the test loader.
    :showactual: Specifies whether actual keypoints should be plotted 
                 in addition to predicted keypoints.
    :figsize: The size of the plot.
    :markersize: The size of markers used for keypoint coordinates.
    """
    
    f, axs = plt.subplots(plotrows, plotcolumns, figsize=figsize)
    model = model.to(device)
    
    # Convert test_loader into an iterator
    test_loader = iter(test_loader)
    
    # set up function for plotting keypoints
    pointsplot = lambda axiz, pts, color: axiz.scatter(
        pts[:, 0], 
        pts[:, 1], 
        s=markersize, 
        marker='.', 
        c=color
    )
    
    if len(axs.shape) == 1: axs = axs.reshape(1, -1)
    
    for ax_ in axs:
        
        # > Get next batch of images and keypoints
        # > Convert images to FloatTensors
        # > Obtain model predictions for image
        # > Flatten keypoints
        images, gt_pts = test_loader.next()
        images = images.type(fltTensor)
        output_pts = model(images)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        for i, ax in enumerate(ax_):
            
            # Convert image to numpy image
            # and convert it to numpy image format
            
            if torch.cuda.is_available():   
                images = images.cpu()
                output_pts = output_pts.cpu()
                
            image = images[i].data.numpy()
            image = np.transpose(image, (1, 2, 0))

            # Remove transformations from predicted keypoints
            prediction = output_pts[i].data.numpy() * 50.0 + 100

            # Plot predicted keypoints on image
            ax.imshow(np.squeeze(image), cmap='gray')
            pointsplot(ax, prediction, 'm')
            
            # plot ground truth points as green pts
            if showactual:
                actual_keypts = gt_pts[i] * 50.0 + 100
                pointsplot(ax, actual_keypts, 'g')

            ax.axis('off')

    plt.show()
