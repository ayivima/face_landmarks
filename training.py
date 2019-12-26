"""Implements a function for training model."""


import torch


__author__ = "Victor Mawusi Ayi <ayivima@hotmail.com>"


def fit(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs=1,
    dynamic_lr=False
):
    
    rate_switch=0
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        
        running_loss = 0.0
        model.train()
        
        print("Started Training")
        
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
            
            
            if batch_i % 100 == 99:
                avg_running_loss = running_loss/100
                # Print average loss at end of every 100 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(
                    epoch+1, batch_i+1, avg_running_loss
                ))
                
                running_loss = 0.0
                
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
        
        if dynamic_lr:
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
            if total_batches == 100: break
        
        avg_test_loss = total_test_loss / total_batches    
        print('---\n Average Validation Loss after Epoch {}: {}\n---'.format(
            epoch + 1, avg_test_loss
        ))
        
        test_losses.append(avg_test_loss.item())

    print('Finished Training')
    
    return train_losses, test_losses
