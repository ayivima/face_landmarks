"""Implements a function for training model."""


import torch
from math import inf


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


def fit(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs=1,
    dynamic_lr=False,
    model_save=False,
    save_name="bestmodela.pt"
):
    """Trains a neural network and returns the lists of training 
    and validation losses.
    
    Arguments
    ---------
    :model: Model to be trained
    :criterion: The Loss function
    :optimizer: The optimizer to be used for gradient descent
    :train_loader: A generator for loading training data
    :test_loader: A generator for loading testing data
    :epochs: The number of complete passes through training data
    :dynamic_lr: Specifies whether learning rate gets changed 
                 dynamically during training
    :model_save: Specifies whether best model should be saved,
                 and based on the lowest validation loss.
    :save_name: Specifies the name to be used to save best model
    """
    
    rate_switch=0
    train_losses, test_losses = [], []
    if model_save: min_val_loss = inf
    print("Started Training")
    
    for epoch in range(epochs):
        
        running_loss = 0.0
        model.train()
                
        for batch_i, data in enumerate(train_loader):

            # Get and prepare images and 
            # their corresponding keypoints
            images, key_pts = data

            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # Forward Pass
            output_pts = model(images)
            
            # Backpropagation
            loss = criterion(output_pts, key_pts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print loss statistics
            # and implement learning rate change
            running_loss += loss.item()
            
            
            #if batch_i % 346 == 345:
        batch_num = batch_i + 1
        avg_running_loss = running_loss/batch_num
        # Print average loss at end of all 346 batches
        print('Epoch: {}, Batch Count: {}, Avg. Training Loss: {}'.format(
            epoch+1, batch_num, avg_running_loss
        ))


        # Implement learning rate change dynamically
        if dynamic_lr:
            if avg_running_loss<0.04 and rate_switch==0:
                optimizer.param_groups[0]['lr']=1e-4
                rate_switch=1
            elif avg_running_loss<0.035 and rate_switch<2:
                optimizer.param_groups[0]['lr']=1e-5
                rate_switch=2
            elif avg_running_loss<0.030 and rate_switch<3:
                optimizer.param_groups[0]['lr']=1e-10
                rate_switch=3
        
            print("Learning Rate:", optimizer.param_groups[0]['lr'])
        
        train_losses.append(avg_running_loss)
        
        # =============================================
        # Get Average Loss on a subset of Training data
        # =============================================
        
        model.eval()
        total_batches = 0
        total_test_loss = 0
        for images, key_pts in test_loader:

            total_batches += 1

            key_pts = key_pts.view(key_pts.size(0), -1)
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # Forward Pass
            output_pts = model(images)
            
            # Backpropagation
            loss = criterion(output_pts, key_pts)
            total_test_loss += loss
            
            # Break at the 100th image, keypoints pair
            if total_batches == 200: break
        
        avg_val_loss = total_test_loss / total_batches    
        print('\t Average Validation Loss: {}'.format(avg_val_loss))
        
        avg_val_loss_item = avg_val_loss.item()
        test_losses.append(avg_val_loss_item)
        
        if model_save:
            if avg_val_loss_item < min_val_loss:
                min_val_loss = avg_val_loss_item
                torch.save(model.state_dict(), save_name)

    print('Finished Training')
    
    return train_losses, test_losses
