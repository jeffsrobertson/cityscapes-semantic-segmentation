from utils import *
import torch
from torch import nn
import numpy as np
import time
from collections import OrderedDict


def test(index, dataset, model):
    """Runs a prediction on a single image from the provided dataset. Plots the original image, ground truth
        segmentation labels, and the predicted segmentation from the model.

    index: (int) Index of the image from the dataset that you want to plot.
    dataset: (Pytorch dataset) Cityscapes TEST dataset, loaded from torchvision.
    model: (Pytorch model) Trained neural network for semantic segmentation.
    """

    # Load raw  and segmented images
    img, seg = dataset[index]
    class_ids = dataset.class_ids

    # Run through model
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        yhat = model(img).squeeze(0)  # dim (n_classes, height, width)
    prediction = yhat.argmax(axis=0)  # dim (height, width)

    # Have to trim edges off image, to match size of prediction
    img = centercrop(img, size=prediction.shape)
    img = img.squeeze(0)
    img = np.transpose(np.array(img), axes=(1, 2, 0))
    pred = np.array(prediction.cpu())
    seg = seg.cpu().numpy()

    # Plot raw and segmented images stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    plt.subplots_adjust(hspace=0)
    plt1 = ax1.imshow(img)
    plt2 = ax2.imshow(seg, cmap='jet', vmin=0, vmax=33)
    # ax2.text(200, 100, 'test', color='white')
    plt3 = ax3.imshow(pred, cmap='jet', vmin=0, vmax=33)
    ax1.set_yticklabels([])
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    ax1.set_ylabel('Image')
    ax2.set_ylabel('Ground truth')
    ax3.set_ylabel('Prediction')

    # Add a colorbar with corresponding seg labels
    if class_ids is not None:
        seg_ids = np.unique(seg)
        cbar = fig.colorbar(plt2, ax=[ax1, ax2, ax3])
        labels = [class_ids[i] for i in seg_ids]
        cbar.set_ticks(seg_ids)
        cbar.set_ticklabels(labels)


def train(model, dataloader, criterion, optimizer):
    model.train()
    acc, class_ious, loss = _run_model(model, dataloader, criterion, optimizer=optimizer, mode='train')
    return acc, class_ious, loss


def validate(model, dataloader, criterion):
    model.eval()
    with torch.no_grad():
        acc, class_ious, loss = _run_model(model, dataloader, criterion, optimizer=None, mode='val')
    return acc, class_ious, loss


def _run_model(model, dataloader, criterion, optimizer=None, mode='train'):
    if type(model) == nn.DataParallel:
        num_classes = model.module.n_classes
    else:
        num_classes = model.n_classes

    batch_time = AverageMeter()
    losses = AverageMeter()
    class_IoUs = ClassIoUMeter(input_size=(1, num_classes))
    accuracy = AverageMeter()

    start_time = time.time()
    for i, (x, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            x = x.to("cuda")
            y = y.to("cuda")

        yhat = model(x)
        loss = criterion(yhat, y.long())
        new_class_ious = calculate_IoU(yhat, y)
        new_acc = calculate_accuracy(yhat, y)

        # Backprop
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update tracking parameters
        batch_time.update(time.time() - start_time, n=x.shape[0])
        losses.update(loss.item(), n=x.shape[0])
        class_IoUs.update(new_class_ious)
        accuracy.update(new_acc, n=x.shape[0])

        # Print progress
        if mode == 'train':
            print(
                '{num_batch}/{total_batches} batches complete. train_loss: {loss:.3f}. train_acc: {acc:.3f}. train_iou: {iou:.3f}. Time: {batch_time:.1f} sec. '.format(
                    num_batch=i + 1,
                    total_batches=len(dataloader),
                    batch_time=batch_time.val,
                    loss=losses.avg,
                    acc=accuracy.avg, iou=class_IoUs.average_iou()), end='\r')
        else:
            print('Evaluating model. {num_batch}/{total_batches} batches complete.'.format(
                num_batch=i + 1,
                total_batches=len(dataloader)), end='\r')

    avg_acc = accuracy.avg
    avg_class_ious = class_IoUs.avg
    avg_loss = losses.avg
    print('')
    return avg_acc, avg_class_ious, avg_loss


def _conv_x2_block(in_features, out_features, padding):
    """Returns two convolutional layers, each backed by a batch normalization layer and ReLU activation"""
    pad = 1 if padding == 'same' else 0

    return nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=3, padding=pad),
                         nn.BatchNorm2d(out_features),
                         nn.ReLU(),
                         nn.Conv2d(out_features, out_features, kernel_size=3, padding=pad),
                         nn.BatchNorm2d(out_features),
                         nn.ReLU())


def _upsample_block(in_features, out_features):
    """Returns a transposed convolutional layer, followed by a batch norm and ReLU activation"""
    return nn.Sequential(nn.ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2),
                         nn.BatchNorm2d(out_features),
                         nn.ReLU())


class UNET(nn.Module):
    def __init__(self, n_classes, padding='valid'):
        super().__init__()
        self.n_classes = n_classes
        self.padding = padding

        # ENCODING PHASE
        self.conv1 = _conv_x2_block(3, 64, padding)
        self.conv2 = _conv_x2_block(64, 128, padding)
        self.conv3 = _conv_x2_block(128, 256, padding)
        self.conv4 = _conv_x2_block(256, 512, padding)
        self.conv5 = _conv_x2_block(512, 1024, padding)

        # DECODING PHASE
        self.upsample6 = _upsample_block(1024, 512)
        # Concat here
        self.conv6 = _conv_x2_block(1024, 512, padding)

        self.upsample7 = _upsample_block(512, 256)
        # Concat here
        self.conv7 = _conv_x2_block(512, 256, padding)

        self.upsample8 = _upsample_block(256, 128)
        # Concat here
        self.conv8 = _conv_x2_block(256, 128, padding)

        self.upsample9 = _upsample_block(128, 64)
        # Concat here
        self.conv9 = _conv_x2_block(128, 64, padding)

        # Classification layer
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)

        # Reused layers
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=.25)

    def forward(self, x):

        # ENCODING PHASE
        x1 = self.conv1(x)
        x = self.pool(x1)
        x = self.dropout(x)

        x2 = self.conv2(x)
        x = self.pool(x2)
        x = self.dropout(x)

        x3 = self.conv3(x)
        x = self.pool(x3)
        x = self.dropout(x)

        x4 = self.conv4(x)
        x = self.pool(x4)
        x = self.dropout(x)

        # BRIDGE
        x = self.conv5(x)

        # DECODING PHASE
        x = self.upsample6(x)

        x = self.dropout(x)
        x = torch.cat([x, centercrop(x4, size=x.shape[2:])], dim=1)
        x = self.conv6(x)

        x = self.upsample7(x)
        x = self.dropout(x)
        x = torch.cat([x, centercrop(x3, size=x.shape[2:])], dim=1)
        x = self.conv7(x)

        x = self.upsample8(x)
        x = self.dropout(x)
        x = torch.cat([x, centercrop(x2, size=x.shape[2:])], dim=1)
        x = self.conv8(x)

        x = self.upsample9(x)
        x = self.dropout(x)
        x = torch.cat([x, centercrop(x1, size=x.shape[2:])], dim=1)
        x = self.conv9(x)

        # Classification layer
        x = self.classifier(x)

        return x

    def load_state_dict(self, state_dict):
        """
        Models that were trained on parallel GPUs need this special workaround so they can be used on single GPU/CPU.
        """
        try:

            super().load_state_dict(state_dict)
            print('this worked')
        except:
            try:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]
                    new_state_dict[name] = v
                super().load_state_dict(new_state_dict)
                print('take 2 worked')
            except:
                raise ValueError("Unable to load weights to model.")