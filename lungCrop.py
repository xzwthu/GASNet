import SimpleITK as sitk 
import numpy as np
from lungmask import mask
from skimage import measure
from scipy.ndimage import zoom
import glob
import os 


class LungMaskCrop():
    def __init__(self):
        self.model = mask.get_moel('unet','R231CovidWeb')
        
    def crop(self,imagepath,labelpath=None):
        
        input_image = sitk.ReadImage(imagepath)
        image_array = sitk.GetArrayFromImage(input_image)
        try:
            segmentation = mask.apply(input_image, model = model1,batch_size=3)
        except:
            print(imagepath)
        labels = measure.label(segmentation)
        if labels.max()>2:
            props = measure.regionprops(labels)
            for i in range(len(props)):
                if props[i].area/512/512/segmentation.shape[0]<0.001:
                    labels[labels==props[i].label] =0
        labels[labels>0]=1
        props = measure.regionprops(labels)
        xmin,ymin,zmin,xmax,ymax,zmax = props[0].bbox
        image_array = image_array[xmin:xmax,ymin:ymax,zmin:zmax]
        segmentation = labels[xmin:xmax,ymin:ymax,zmin:zmax]
        x,y,z = image_array.shape
        image_array = zoom(image_array,(40/x,160/y,160/z),order=1)
            
        # normalization
        image_array = image_array+1250
        image_array[image_array<0]=0
        image_array[image_array>1500]=1500
        image_array = image_array/1500*255

        segmentation = zoom(segmentation,(40/x,160/y,160/z),order=0)
        segmentation = segmentation.astype('uint8')
        
        image_image = sitk.GetImageFromArray(image_array)
        seg_image  = sitk.GetImageFromArray(segmentation)
        if labelpath is not None:
            label_image = sitk.ReadImage(labelpath)
            label_array = sitk.GetArrayFromImage(label_image)
            label_seg = label_array[xmin:xmax,ymin:ymax,zmin:zmax]
            label_image = sitk.GetImageFromArray(label_seg.astype('uint8'))
            return image_image,seg_image,label_image
        return image_image,seg_image

if __name__ == "__main__":
    LungMask = LungMaskCrop()
    imagepath = 'path/to/image'
    labelpath = 'path/to/label'
    crop_image,crop_lung,crop_lesion = LungMask.crop(imagepath,labelpath=labelpath)
