import numpy as np
import matplotlib.pyplot as plt

def pixelwise_loss_old(yhat, y):
    """
    UPDATE: After comparing a bunch of unit tests between this and the built-in nn.CrossEntropyLoss, I confirmed the
    latter is calculating the loss correctly. I wasn't confident initially that it would correctly handle the 4D tensors
    from a segmentation NN.
    
    Calculate the pixel-avg'd cross-entropy loss for a 2-D semantic segmentation prediction.
    Note that the prediction (yhat) tensor should NOT be softmaxed! i.e. it should be whatever the last ReLU layer spits out
    
    Input:
        yhat: (Tensor) Predicted segmentation of model, of shape (batch, num_features, height, width). The dimension
            num_features is a softmax classification output, whose length corresponds to the number of possible labels.
        y: (Tensor) Ground truth segmentation label, of shape, (batch, height, width). Each value in tensor is
            an int corresponding to the label for that pixel.
            
    Returns:
        loss: (FloatTensor) 0D tensor corresponding to the pixel-averaged, batch-averaged cross-entropy loss.
    
    """
    
    # If valid padding was used, we need to trim edges of ground truth to match prediction 
    y = centercrop(y, size=yhat.shape[-2:])
    
    # Get predicted values from yhat, at indices corresponding to ground truth label
    index = y.unsqueeze(1).long()
    num_nans = torch.isnan(yhat.view(-1)).sum().item()
    predictions = yhat.gather(dim=1, index=index).squeeze(1)
    
    # Calculate cross-entropy of the softmaxed classification layer
    # Sum over pixels, average over batches
    num_batches = predictions.shape[0]
    num_pixels = predictions.shape[1]*predictions.shape[2]
    loss = -(1./(num_batches*num_pixels))*torch.log(predictions).sum()
    
    return loss

def pixelwise_loss(yhat, y):
    """
    Calculate the pixel-wise cross-entropy loss for a 2-D semantic segmentation prediction.
    Note that the prediction (yhat) tensor should NOT be softmaxed! i.e. it should be whatever the last ReLU layer spits out
    
    Input:
        yhat: (Tensor) Predicted segmentation of model, of shape (batch, num_features, height, width). The dimension
            num_features is a softmax classification output, whose length corresponds to the number of possible labels.
        y: (Tensor) Ground truth segmentation label, of shape, (batch, height, width). Each value in tensor is
            an int corresponding to the label for that pixel.
            
    Returns:
        loss: (FloatTensor) 0D tensor corresponding to the pixel-summed, batch-averaged cross-entropy loss.
    
    """
    
    # If valid padding was used, we need to trim edges of ground truth to match prediction 
    y = centercrop(y, size=yhat.shape[-2:])
    
    # Calculate cross-entropy of the softmaxed classification layer
    return nn.CrossEntropyLoss()(yhat, y)

def calculate_accuracy(yhat, y):
    """Computes the "top k" accuracy for the specified values of k. 
            i.e. if k=5, it will calculate the % of pixels where the top 5 predicted labels contain the ground
            truth label.
            
        Input:
            yhat: (Tensor) Predicted segmentation of the model, of shape (batch_size, num_features, height, width).
            y: (Tensor) Ground truth labels for the pixels, of shape (batch_size, height, width). Each value is an 
                integer corresponding to the label for that pixel.
            topk: (tuple) Will calculate the 'top k' for all the values in this tuple.
            
            Note that if y and yhat have different height/widths (i.e. from using valid padding), it will trim 
            the larger image.
            
        Returns:
        
        """
    
    yhat = yhat.detach()
    batch_size = y.shape[0]
    
    # If valid padding was used, we need to trim edges of ground truth to match prediction 
    y = centercrop(y, size=yhat.shape[-2:])

    # pred = index in each batch corresponding to max value
    #_, pred = yhat.topk(maxk, dim=1, largest=True, sorted=True)
    predictions = yhat.argmax(dim=1) # (batch, height, width)
    correct_pixels = torch.eq(predictions, y).sum()
    total_pixels = y.numel()
    return correct_pixels.item()/total_pixels
        

def calculate_IoU(yhat, y):
    """Computes the intersection over union (IoU) for every class. If batch size > 1, calculates the IoU for each sample
        in batch, then averages IoUs together.
            
        Input:
            yhat: (Tensor) Predicted segmentation of the model, of shape (batch_size, num_features, height, width).
            y: (Tensor) Ground truth labels for the pixels, of shape (batch_size, height, width). Each value is an 
                integer corresponding to the label for that pixel.
                
            Note that if y and yhat have different height/widths (i.e. from using valid padding), it will trim 
            the larger image.
            
        Returns:
            iou_list: (2D np.array) Array of shape (num_batches, n_classes), with values corresponding 
            to the IoU for each class in that batch.
        
        """
    
    yhat = yhat.detach()
    batch_size, num_classes, _, _ = yhat.shape
    
    # Debugging
    num_negative = (y == -1).sum()
    if num_negative > 0:
        print('Found {} negative values in target'.format(num_negative.item()))
        print('')
    
    # If valid padding was used, we need to trim edges of ground truth to match prediction 
    y = centercrop(y, size=yhat.shape[-2:])

    # Find indices corresponding to max predicted class
    predictions = yhat.argmax(dim=1)
    
    # Calculate the IoU one at a time for each class
    iou_list = [np.nan]*num_classes
    for ci in range(len(iou_list)):
        y_mask = y == ci
        pred_mask = predictions == ci
        intersection = (pred_mask & y_mask).float().sum(dim=(1, 2))
        union = (pred_mask | y_mask).float().sum(dim=(1, 2))
            
        # Calculate the IoU for each sample, then average across all batches
        # We do this in numpy to handle NaN values (i.e. for samples where there were no pixels for a given class)
        with np.errstate(divide='ignore'):
            batch_ious = (intersection.cpu().numpy()/union.cpu().numpy())
            iou_list[ci] = batch_ious

    return np.array(iou_list).T

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class ClassIoUMeter(object):
    """Tracks the average and current IoU of each class. 
        Similar to AverageMeter except it tracks arrays of values.
    """
    def __init__(self, input_size):
        self.values = None
        self.avg = np.zeros(shape=input_size)
        
    def update(self, new_values):
        
        num_batches, num_classes = new_values.shape
        
        if self.values is None:
            self.values = new_values
        else:
            self.values = np.concatenate((self.values, new_values), axis=0)
            
        self.avg = np.nanmean(self.values, axis=0)
        
    def average_iou(self):
        return np.nanmean(self.avg)
    
def plot_cityscape(index, dataset, class_ids=None):
    """Plots an image from the Cityscapes dataset, along with its segmented image.
    
    index: (int) Index of the image from the dataset that you want to plot.
    dataset: (Pytorch dataset) Cityscapes dataset, loaded from torchvision.
    """
    
    # Load raw  and segmented images
    img, seg = dataset[index]
    img = np.array(img)
    seg = np.array(seg)
    img = np.transpose(img, axes=(1, 2, 0))
    
    # Segmented image contains values > 1000, for instance segmentation. 
    # Reduce these to [0, 33] for semantic segmentation
    inds = np.where(seg >= 1000)
    seg[inds] = seg[inds] // 1000
    
    # Plot raw and segmented images stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    plt.subplots_adjust(hspace=0)
    plt1 = ax1.imshow(img)
    plt2 = ax2.imshow(seg, cmap='jet')
    
    # Add a colorbar with corresponding seg labels
    if class_ids is not None:
        seg_ids = np.unique(seg)
        cbar = fig.colorbar(plt2, ax=[ax1, ax2])
        labels = [class_ids[i] for i in seg_ids]
        cbar.set_ticks(seg_ids)
        cbar.set_ticklabels(labels)

def centercrop(tensor, size):
    """Center crops the given tensor in the height and width dimensions to be the desired size.

    Input:
        tensor: 4d tensor of shape (batch, features, height, width)
        size: 2d tuple of desired (dheight, dwidth)

    Returns:
        new_tensor: cropped 4d tensor of shape (batch, features, dheight, dwidth)
    """
    
    input_dims = tensor.ndim
    if input_dims == 4:
        _, _, old_height, old_width = tensor.shape
    elif input_dims == 3:
        _, old_height, old_width = tensor.shape
    elif input_dims == 2:
        old_height, old_width = tensor.shape
    else:
        raise ValueError("Input tensor for centercrop() must be 2-4 dimensional, instead got tensor of dim {}".format(input_dims))
        
    new_height, new_width = size
    assert old_height >= new_height
    assert old_width >= new_width
    if (old_height == new_height) and (old_width == new_width):
        return tensor

    xslice = slice(int(.5*(old_width - new_width)), int(.5*(old_width + new_width)))
    yslice = slice(int(.5*(old_height - new_height)), int(.5*(old_height + new_height)))
    if input_dims == 4:
        new_tensor = tensor[:, :, yslice, xslice]
    elif input_dims == 3:
        new_tensor = tensor[:, yslice, xslice]
    else:
        new_tensor = tensor[yslice, xslice]
    return new_tensor
