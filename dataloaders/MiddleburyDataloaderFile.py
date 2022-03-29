"""
This file defines the dataloader to load the images from BOSCH DATASET to feed the model
Author:Alejandro
"""

import glob
import numpy as np
from PIL import Image
import os
from pyrsistent import v
import torch
import torch.utils.data as data
import cv2
from torchvision import transforms
from dataloaders.data_augmentation import DataAugmentation
#from dataloaders.data_augmentation import DataAugmentation,DataAugmentation_Albu
from test_occlusions_removal import slow_remove_occlusions


def read_rgb(path):
    #rgb=np.array(cv2.imread(path))/255.0
    rgb = cv2.cvtColor(
        cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)#/255.0
    #print("SHAPE JUST READED RGGB----->"+str(rgb.shape))
    gray = np.array(Image.fromarray(cv2.imread(path)).convert('L'))#/255.0
    gray = np.expand_dims(gray, -1)
    #print("RGB_>"+str(rgb.max()))
    return rgb, gray

def read_mask(path):
    mask=np.load(path)['arr_0']
    mask=np.expand_dims(mask, 2)
    return mask


def read_depth(path, preprocess_depth=False):
    """
    Reads the depth image and change the values from 0-1000 to 0-255 range
    """
    depth_original=np.array(cv2.imread(path,cv2.IMREAD_GRAYSCALE))
    #print("Depth Type->"+str(depth.dtype))
    #depth=np.array(Image.fromarray(depth).convert('L'))#/255
    #print(depth.shape)
    if preprocess_depth:
        depth=slow_remove_occlusions(depth_original)
    depth=np.expand_dims(depth_original,2)

    #print(depth.dtype)
    #print("Depth->"+str(depth.max()))

    return depth


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def ToTensor(img):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W

    Convert a ``numpy.ndarray`` to tensor.

    Args:
        img (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not (_is_numpy_image(img)):
        raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                           str(type(img))+') and dimensions ('+str(img.shape)+')')

    if isinstance(img, np.ndarray):
        # handle numpy array
        if img.ndim == 3:
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
        elif img.ndim == 2:
            img = torch.from_numpy(img.copy())
        else:
            raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                               str(type(img))+') and dimensions ('+str(img.shape)+')')
        return img


def get_paths(train_or_test):
    """Returns the paths for the data files

    Args:
        train_or_test (["train"/"test"]): To load files from train or test directory

    Returns:
        [dict]: {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt, "mask":paths_mask}
    """
    if train_or_test == "train":
        path_gt = "data_middlebury/train/gt/*.png"
        path_rgb = "data_middlebury/train/rgb/*.png"
    elif train_or_test == "test":
        path_gt = "data_middlebury/test/gt/*.png"
        path_rgb = "data_middlebury/test/rgb/*.png"

    paths_rgb = sorted(glob.glob(path_rgb))
    paths_gt = sorted(glob.glob(path_gt))
    if len(paths_rgb) == 0 or len(paths_gt) == 0:
        raise (RuntimeError(
            "ALEJANDRO ERROR--->No images in some or all of the paths of the dataloader.\n Len(RGB)="+str(len(paths_rgb))+"\n Len(gt)="+str(len(paths_gt))+"\n "))
    paths = {"rgb": paths_rgb, 'gt': paths_gt}
    return paths


class MiddleburyDataLoader(data.Dataset):
    def __init__(self, train_or_test, augment=True, preprocess_depth=False):
        """Inits the MiddleburyDataLoader dataloader

        Args:
            train_or_test (string): "train" or "test" to use different data (train and inference)
        """
        self.train_or_test = train_or_test
        self.paths = get_paths(train_or_test)
        self.preprocess_depth=preprocess_depth
        self.data_aug=DataAugmentation(size=(512,512))
        self.augment=augment


    def __getraw__(self, index):
        rgb, gray = read_rgb(self.paths['rgb'][index]) if (
            self.paths['rgb'][index] is not None) else None
        gt = read_depth(self.paths['gt'][index]) if (
            self.paths['gt'][index] is not None) else None
        return rgb, gray, gt

    def __getitem__(self, index):
        rgb, gray, gt = self.__getraw__(index)
        
        
        
        DIAMOND_KERNEL_15 = np.asarray(
        [
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        ], dtype=np.uint8)
        
        mask=(gt==0).astype(np.uint8)
        degraded_mask = cv2.dilate(mask, DIAMOND_KERNEL_15)   
        degraded_mask=np.expand_dims(degraded_mask,2)
        degraded_depth=(1-degraded_mask)*gt
                
                
        items = {"rgb": rgb, "d": degraded_depth, 'gt':gt, 'g':gray}
    
        h=512
        w=512
        
        resize=transforms.Resize((h,w))    
        preprocess_rgb=transforms.Compose([
            transforms.Resize((h,w)),
            #transforms.Normalize(mean=[0.4409, 0.4570, 0.3751], std=[0.2676, 0.2132, 0.2345])
            #transforms.Normalize(mean=[0, 0, 0], std=[1,1,1])          
        ])
        preprocess_depth=transforms.Compose([
            transforms.Resize((h,w)),
            #transforms.Normalize(mean=[0.2674], std=[0.1949])  
            #transforms.Normalize(mean=[0], std=[1])          
        
        ])
        preprocess_gt=transforms.Compose([
            transforms.Resize((h,w)),
            #transforms.Normalize(mean=[0.3073], std=[0.1761])
            #transforms.Normalize(mean=[0], std=[1])             
        ])
        #print("init rgb shape->"+str(rgb.shape))
        rgb=ToTensor(rgb).float()/255
        #print("med rgb shape->"+str(rgb.shape))
        sparse=ToTensor(degraded_depth).float()/255
        gt=ToTensor(gt).float()/255
        gray=ToTensor(gray).float()/255

        #sparse=preprocess_depth(sparse)
 

        #items={'rgb':preprocess_rgb(rgb), 'd':sparse, 'gt':preprocess_gt(gt), 'g':resize(gray), 'occlusion_mask':occlusion_mask}
        items={'rgb':preprocess_rgb(rgb), 'd':preprocess_depth(sparse), 'gt':preprocess_gt(gt), 'g':preprocess_depth(gray)}
        
        
        if self.augment:
            rgb, sparse, gt, gray=self.data_aug(items)
            items = {"rgb": rgb, "d": sparse, 'gt':gt, 'g':gray}
        
    
        return items

    def __len__(self):
        return len(self.paths['rgb'])
