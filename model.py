from utils import *
import torch
from torch import nn
import numpy as np

def conv_x2_block(in_features, out_features, padding):
    """Returns two convolutional layers, each backed by a batch normalization layer and ReLU activation"""
    pad = 1 if padding == 'same' else 0
    
    return nn.Sequential(nn.Conv2d(in_features, out_features, kernel_size=3, padding=pad), 
                           nn.BatchNorm2d(out_features), 
                           nn.ReLU(), 
                           nn.Conv2d(out_features, out_features, kernel_size=3, padding=pad), 
                           nn.BatchNorm2d(out_features), 
                           nn.ReLU())

def upsample_block(in_features, out_features):
    return nn.Sequential(nn.ConvTranspose2d(in_features, out_features, kernel_size=2, stride=2), 
                   nn.BatchNorm2d(out_features), 
                   nn.ReLU())

class UNET(nn.Module):
    def __init__(self, n_classes, padding='valid'):
        super().__init__()
        self.n_classes = n_classes
        self.padding = padding

        # ENCODING PHASE
        self.conv1 = conv_x2_block(3, 64, padding)
        self.conv2 = conv_x2_block(64, 128, padding)
        self.conv3 = conv_x2_block(128, 256, padding)
        self.conv4 = conv_x2_block(256, 512, padding)
        self.conv5 = conv_x2_block(512, 1024, padding)
        
        # DECODING PHASE
        self.upsample6 = upsample_block(1024, 512)
        # Concat here
        self.conv6 = conv_x2_block(1024, 512, padding)
        
        self.upsample7 = upsample_block(512, 256)
        # Concat here
        self.conv7 = conv_x2_block(512, 256, padding)
        
        self.upsample8 = upsample_block(256, 128)
        # Concat here
        self.conv8 = conv_x2_block(256, 128, padding)
        
        self.upsample9 = upsample_block(128, 64)
        # Concat here
        self.conv9 = conv_x2_block(128, 64, padding)
        
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
