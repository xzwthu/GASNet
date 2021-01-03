# GASNet
This is a PyTorch implementation of GASNet for lesion segmentation of COVID-19 in a weakly-supervised scenario, source code of the paper: [GASNet: Weakly-supervised Framework for COVID-19 Lesion Segmentation](https://arxiv.org/abs/2010.09456?context=cs.CV). 

## Architecture
![GASNet](./pics/pipeline.png "pipeline of GASNet")
Three modules with optimizable parameters compose the framework of GASNet, the segmenter (S), the generator (G), and the discriminator (D). 
    The segmenter (S) predicts the infected areas given a CT volume. The generator (G) generates
    a new volume. By replacing lesion areas in the original volume by the generated volume,
    a synthetic volume is obtained. 
    The synthetic volume and the real healthy volume are input into a discriminator (D) for classification. The parameters of the G and the S are
      optimized to make the synthetic volume look as similar as possible to a real healthy volume, so that the discriminator can not distinguish. 
      Original COVID-19 volume and the generated volume are also fed into the D as auxiliary constraints. 
      Only the S part is needed during the test, 
      which means GASNet has no extra computational 
      burden compared with the standard segmentation network. 

## Prerequisites
- python 3.7
- pytorch 1.2.0
- torchvision 0.4.0
- numpy, skimage, scipy, SimpleITK, and imgaug

## Usage 
### Data-preprocessing 
Your need to do some data-preprocessing work before using the data to train GASNet. An open-soure lung segmentation model is available at <https://github.com/JoHof/lungmask>, and you can use it to get the bound box around the lung area. The 3D volume sample of each CT, along with the lung mask and lesion segmentation label (if any),
is cropped along the lung area. The cropped CT volume is then
resized into 40×160×160, and its value is clipped into [-1250,250]. 

lungCrop.py is a script code for preprocessing. You can directly call the LungMaskCrop class to pipelining your data, and then store the results in the corresponding folder directory.

To run the code both in training and testing, three dirs are needed, including *./images* to store CT volumes, *./lung* to store lung segmentation results, and *./lesion* to store lesion labels. Change the root path of the dataset in *./configs/Gan.yaml* to your dataset path and modify the splition configuration according to your needs. 

### Training
Simply run 
```
sh train.sh
```
to train GASNet. To add or delete the constraint losses metioned in the paper, you can change the prefix of EXP.ID in the configure file.

### Testing
Simply change the path to your trained model and run 
```
sh test.sh
```
to test the performance of GASNet. 

* Because the input of the test code is a cropped image, the size of the segmentation result is 160 × 160 × 40. If you want to get the segmentation result of the same size as the original image, you can run *post_processing.py*. You need to modify the storage paths of the original image and the GASNet segmentation result in the code.
### Result and visualization
- Dataset-A consists of 20 CT volumes and 10 of them have been transformed to the range of [0, 255]. Considering the original CT values are unavailable, we divide them into two subsets and test the performances respectively.
 The dataset is available at [CovidSeg](https://zenodo.org/record/3757476#.X41Jj-biuiN)
- Dataset-B contains 9 COVID-19 CT volumes. 
The dataset is available at [MedSeg](https://medicalsegmentation.com/covid19/)
- Dataset-C and Dataset-D (Volume-level annotation) are from MosMed. which consists of 856 CT volumes
with COVID-19 related findings as well as 254 CT volumes
without such findings. 50 COVID-19 cases have voxel-level
annotation of lesions by experts, which forms Dataset-C. The
rest of the data, consisting of 254 healthy volumes and 806
COVID-19 volumes excluding 50 voxel-level labeled samples,
forms Dataset-D. The diagnosis results of the CT volumes can
be used as volume-level labels directly.
The dataset is available at [MosMed](https://mosmed.ai/en/)
- Dataset-E (Volume-level annotation) is a large dataset
with volume-level annotation we collected, in which 1,678
COVID-19 CT volumes come from the Wuhan Union Hospital, whose patients have been diagnosed as COVID-19 positive
by nucleic acid testing, and 1,031 healthy CT volumes come
from the routine physical examination.

We trained GASNet based on Dataset-E and one voxel-level labeled sample from Dataset-A and test the performance on the rest of the three public dataset with lesion annotation. We then replaced the Dataset-E with Dataset-D as our volume-level annotation dataset and finetuned the model. The Dice scores of these two trained models are shown below:
|                 | Dataset-A_sub1 | Dataset-A_sub2 | Dataset-B    | Dataset-C    |
| --------------- | -------------- | -------------- | ------------ | ------------ |
| GASNet          | 76.7±6.1(%)   | 63.2±19.4(%)  | 60.2±23.4(%) | 54.2±22.4(%) |
| GASNet_finetune | -              | -              | 59.7±18.5(%) | 58.9±24.4(%) |

For detailed results of our experiments and comparison between GASNet and other existing methods on COVID-19 lesion segmentation, please refer to our paper.
![vis2](./pics/vis2.png)
## Citation
Please remember to cite our article if you find it helpful to your work
```
@misc{xu2020gasnet,
      title={GASNet: Weakly-supervised Framework for COVID-19 Lesion Segmentation}, 
      author={Zhanwei Xu and Yukun Cao and Cheng Jin and Guozhu Shao and Xiaoqing Liu and Jie Zhou and Heshui Shi and Jianjiang Feng},
      year={2020},
      eprint={2010.09456},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## Contact
Never mind opening issues about the code or data. If you have any further questions about GASNet, please contact me <xzw14@tsinghua.org.cn>

