import torch
from torchvision.datasets import Cityscapes
from torchvision.transforms import Compose, RandomRotation, ToTensor, Resize, RandomHorizontalFlip
import torchvision.transforms.functional as TF
import numpy as np

class Cityscapes(Cityscapes):
    def __init__(self, root, split='train', mode='fine'):
        super().__init__(root=root, split=split, mode=mode, target_type='semantic')

        self.class_ids = {0: 'unlabeled',
             1: 'ego vehicle',
             2: 'rectification border',
             3: 'out of roi',
             4: 'static',
             5: 'dynamic',
             6: 'ground',
             7: 'road',
             8: 'sidewalk',
             9: 'parking',
             10: 'rail track',
             11: 'building',
             12: 'wall',
             13: 'fence',
             14: 'guard rail',
             15: 'bridge',
             16: 'tunnel',
             17: 'pole',
             18: 'polegroup',
             19: 'traffic light',
             20: 'traffic sign',
             21: 'vegetation',
             22: 'terrain',
             23: 'sky',
             24: 'person',
             25: 'rider',
             26: 'car',
             27: 'truck',
             28: 'bus',
             29: 'caravan',
             30: 'trailer',
             31: 'train',
             32: 'motorcycle',
             33: 'bicycle'
            }
        
    def __getitem__(self, i):
        img, seg = super().__getitem__(i)
        
        rescale = Compose([Resize(size=(int(.5*img.size[1]), int(.5*img.size[0]) ))])
        
        # Reduce size of image for reduced GPU memory usage
        img = rescale(img)
        seg = rescale(seg)
        
        # Apply data augmentation
        if self.split == 'train':
            if np.random.random() > .5:
                img = TF.hflip(img)
                seg = TF.hflip(seg)
                
        # Convert img to tensor, normalize
        img = ToTensor()(img) # This normalizes pixels to be in range [0, 1]
        seg = torch.from_numpy(np.array(seg))
        
        # License plates are labelled by a "-1" - reassign these to the label 0
        seg[seg < 0] = 0
        
        return img, seg
    