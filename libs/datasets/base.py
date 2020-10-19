import random
import torch 
from PIL import Image
from torch.utils import data 
from torchvision import transforms
import os.path as osp
import numpy as np
import SimpleITK as sitk
class myDataset(data.Dataset):
    def __init__(
        self,
        root,
        split,
        ignore_label,
        crop_size,
        augment = True,
    ):
        self.root = root
        self.split = split
        self.ignore_label = ignore_label
        self.augment = augment
        self.crop_size = crop_size
        self.ids = []
        self._set_files()
        
    def _set_augment(self,augment=False):
        self.augment = augment
    def _set_image_dir(self,image_dir = 'images'):
        self.image_dir = osp.join(self.root,image_dir)
    def _set_files(self):
        self.image_dir = osp.join(self.root,'images')
        self.label_dir = osp.join(self.root,'leision')
        self.mask_dir = osp.join(self.root+'lung')
        # if self.split in ['train_id_L','train_id_wL','val_id','train_id_H']:
        file_list = osp.join(
            self.root,'id_txt',self.split +'.txt'
        )
        with open(file_list) as f:
            contents = f.readlines()
        self.ids = [n[:-1] for n in contents]
        # else:
        #     raise ValueError('Invalid split name {}'.format(self.split))
    def _augmentation(self,image,label):
        d,h,w = image.shape
        start_d = random.randint(0,d-self.crop_size/4)
        end_d = int(start_d+self.crop_size/4)
        # import pdb; pdb.set_trace()
        image = image[start_d:end_d,:,:]
        noise_ = np.random.random(image.shape)*0.01-0.005
        image += noise_
        if label is not None:
            label = label[start_d:end_d,:,:]
        return image, label
    def __getitem__(self, index):
        label = None
        image_id = self.ids[index]
        image_path = osp.join(self.image_dir, image_id + '.nii')
        img = sitk.ReadImage(image_path)
        image = sitk.GetArrayFromImage(img)

        # image = np.load(image_path)
        if 'H' in self.split:
            label = np.zeros(image.shape).astype('uint8')
        elif ('_L' in self.split) or ('_naive' in self.split):
            label_path = osp.join(self.label_dir,image_id+'.nii')
            img = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(img)
            # label = np.load(label_path)
        elif 'val' in self.split:
            label_path = osp.join(self.label_dir,image_id+'.nii')
            img = sitk.ReadImage(label_path)
            label = sitk.GetArrayFromImage(img)
            # label = np.load(label_path)
        # import pdb; pdb.set_trace()
        image = image.astype(np.float32)/2624.
        if self.augment:
            image, label = self._augmentation(image,label)
        image = image[np.newaxis,:,:]
        image[image>1]=1
        image[image<0]=0
        image = image*2-1
        if label is not None:
            return image_id,image,label
        else:
            return image_id, image
    def __len__(self):
        return len(self.ids)
    
        
    
