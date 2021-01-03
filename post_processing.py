import SimpleITK as sitk 
import numpy as np
from lungmask import mask
from skimage import measure
from scipy.ndimage import zoom
import glob
import os 
from metric import scores
image_dir = 'path/to/your/original/images'
lesion_dir = 'path/to/the/lesion/prediction/by/GASNet'
save_dir = 'path/to/your/save/dir'
filelist = glob.glob(image_dir)
model2 = mask.get_model('unet','LTRCLobes')
model1 = mask.get_model('unet','R231CovidWeb')
for imagepath in filelist:
    if 1:
        imagename = imagepath.split('/')[-1]
        input_image = sitk.ReadImage(imagepath)
        image_array = sitk.GetArrayFromImage(input_image)
        try:
            segmentation = mask.apply(input_image, model = model1,batch_size=3)
        except:
            print(imagepath)
            import pdb; pdb.set_trace()
        labels = measure.label(segmentation)
        if labels.max()>2:
            props = measure.regionprops(labels)
            for i in range(len(props)):
                if props[i].area/512/512/segmentation.shape[0]<0.001:
                    labels[labels==props[i].label] =0
        labels[labels>0]=1
        props = measure.regionprops(labels)
        xmin,ymin,zmin,xmax,ymax,zmax = props[0].bbox
        image_array_part = image_array[xmin:xmax,ymin:ymax,zmin:zmax]
        x,y,z = image_array_part.shape
        lesion_image = sitk.ReadImage(lesion_dir+imagename)
        lesion_array = sitk.GetArrayFromImage(lesion_image)
        lesion_array_part = zoom(lesion_array,(x/40,y/160,z/160),order=0)
        lesion_array = np.zeros(image_array.shape)
        lesion_array[xmin:xmax,ymin:ymax,zmin:zmax] = lesion_array_part
        lesion_out = sitk.GetImageFromArray(lesion_array)
        lesion_out.CopyInformation(input_image)
        sitk.WriteImage(lesion_out,save_dir+imagename)


