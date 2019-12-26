"""
Implements functions for displaying keypoints on images.
"""


import torch
import matplotlib.pyplot as plt
import numpy as np


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"

        
def plotter(
    model, 
    test_loader, 
    plotrows=5, 
    plotcolumns=8, 
    showactual=False, 
    batch_size=10
):
    
    f, axs = plt.subplots(plotrows, plotcolumns, figsize=(17,10))
    
    # Convert test_loader into an iterator
    test_loader = iter(test_loader)
    
    # set up function for plotting keypoints
    pointsplot = lambda axiz, pts, color: axiz.scatter(
        pts[:, 0], 
        pts[:, 1], 
        s=20, 
        marker='.', 
        c=color
    )
    
    for ax_ in axs:
        
        # > Get next batch of images and keypoints
        # > Convert images to FloatTensors
        # > Obtain model predictions for image
        # > Flatten keypoints
        images, gt_pts = test_loader.next()
        images = images.type(torch.FloatTensor)
        output_pts = model(images)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        for i, ax in enumerate(ax_):
            
            # Convert image to numpy image
            # and convert it to numpy image format
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
