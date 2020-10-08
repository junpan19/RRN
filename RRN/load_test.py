import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from Gaussian_downsample import gaussian_downsample
from bicubic import imresize

def modcrop(img,scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    return img

class DataloadFromFolderTest(data.Dataset): # load test dataset
    def __init__(self, image_dir, scale, scene_name, transform):
        super(DataloadFromFolderTest, self).__init__()
        alist = os.listdir(os.path.join(image_dir, scene_name))
        alist.sort()
        self.image_filenames = [os.path.join(image_dir, scene_name, x) for x in alist] 
        self.L = len(alist)
        self.scale = scale
        self.transform = transform # To_tensor
    def __getitem__(self, index):
        target = []
        for i in range(self.L):
            GT_temp = modcrop(Image.open(self.image_filenames[i]).convert('RGB'), self.scale)
            target.append(GT_temp)
        target = [np.asarray(HR) for HR in target] 
        target = np.asarray(target)
        if self.scale == 4:
            target = np.lib.pad(target, pad_width=((0,0), (2*self.scale,2*self.scale), (2*self.scale,2*self.scale), (0,0)), mode='reflect')
        t, h, w, c = target.shape
        target = target.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT']
        if self.transform:
            target = self.transform(target) # Tensor, [CT',H',W']
        target = target.view(c,t,h,w)
        LR = gaussian_downsample(target, self.scale) # [c,t,h,w]
        LR = torch.cat((LR[:,1:2,:,:], LR), dim=1)
        return LR, target
        
    def __len__(self):
        return 1 

