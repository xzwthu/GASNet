from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os
from time import time as time

import click
import numpy as np
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
import cv2
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm
import shutil
from libs.datasets.base2 import myDataset
from libs.models.model import Generator,Generator3,Discriminator,GaussianBlur,Laplace,Discriminator2
from libs.models.model_3D import UNet3D,NestedUNet,UNet3D_shallow,UNet3D_orig
from libs.models.VNet import VNet
from libs.models.pamr import PAMR
from libs.loss.myloss import myLoss, myLoss2, myLoss3, myLossD
from libs.utils import DenseCRF, scores,score_p,score_p2, get_scheduler, lbl2bgr
import SimpleITK as sitk
import scipy.ndimage as ndimage
import random
def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option(
    "-v",
    "--is-validation",
    is_flag=True,
    help="Performing validation or not [default: not]"
)

def train(config_path,cuda,is_validation):
    CONFIG = Dict(yaml.load(config_path, Loader=yaml.FullLoader))
    print("Exp: ", CONFIG.EXP.ID)

    save_dir = os.path.join("./data/configs", CONFIG.EXP.ID)
    os.makedirs(save_dir, exist_ok=True)
    (_, filename) = os.path.split(config_path.name)
    target_file = os.path.join(save_dir, filename)
    if os.path.abspath(config_path.name) != os.path.abspath(target_file):        
        shutil.copyfile(config_path.name, target_file)

    device = get_device(cuda)
    torch.backends.cudnn.benchmark = True

    # dataset_L
    dataset_L = myDataset(
        root = CONFIG.DATASET.ROOT,
        split = CONFIG.DATASET.SPLIT.LABELED,
        ignore_label = CONFIG.DATASET.IGNORE_LABEL,
        augment = True,
        crop_size = CONFIG.IMAGE.SIZE.TRAIN,
    )
    print(dataset_L)
    loader_L = torch.utils.data.DataLoader(
        dataset=dataset_L,batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,shuffle=False,
    )
    iter_L = iter(loader_L)
    # dataset_WL
    dataset_WL = myDataset(
        root = CONFIG.DATASET.ROOT,
        split = CONFIG.DATASET.SPLIT.WO_LABEL,
        ignore_label = CONFIG.DATASET.IGNORE_LABEL,
        augment = True,
        crop_size = CONFIG.IMAGE.SIZE.TRAIN,
    )
    print(dataset_WL)
    loader_WL = torch.utils.data.DataLoader(
        dataset=dataset_WL,batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,shuffle=True,
    )
    iter_WL = iter(loader_WL)
    # dataset_H
    dataset_H = myDataset(
        root = CONFIG.DATASET.ROOT,
        split = CONFIG.DATASET.SPLIT.HEALTHY,
        ignore_label = CONFIG.DATASET.IGNORE_LABEL,
        augment = True,
        crop_size = CONFIG.IMAGE.SIZE.TRAIN,
    )
    print(dataset_H)
    loader_H = torch.utils.data.DataLoader(
        dataset=dataset_H,batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,shuffle=True,
    )
    iter_H = iter(loader_H)
    #validation
    if is_validation:
        val_dataset = myDataset(
            root = CONFIG.DATASET.ROOT,
            split = CONFIG.DATASET.SPLIT.VALID,
            ignore_label = CONFIG.DATASET.IGNORE_LABEL,
            augment = True,
            crop_size = CONFIG.IMAGE.SIZE.TEST,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
            num_workers=0,#CONFIG.DATALOADER.NUM_WORKERS,
            shuffle=False,
        )
        valwL_dataset = myDataset(
            root = CONFIG.DATASET.ROOT,
            split = CONFIG.DATASET.SPLIT.VALIDWL,
            ignore_label = CONFIG.DATASET.IGNORE_LABEL,
            augment = True,
            crop_size = CONFIG.IMAGE.SIZE.TEST,
        )
        valwL_loader = torch.utils.data.DataLoader(
            dataset=valwL_dataset,
            batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
            num_workers=0,#CONFIG.DATALOADER.NUM_WORKERS,
            shuffle=False,
        )
        valH_dataset = myDataset(
            root = CONFIG.DATASET.ROOT,
            split = CONFIG.DATASET.SPLIT.VALIDH,
            ignore_label = CONFIG.DATASET.IGNORE_LABEL,
            augment = True,
            crop_size = CONFIG.IMAGE.SIZE.TEST,
        )
        valH_loader = torch.utils.data.DataLoader(
            dataset=valH_dataset,
            batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
            num_workers=0,#CONFIG.DATALOADER.NUM_WORKERS,
            shuffle=False,
        )
        iter_valwL = iter(valwL_loader)
        iter_valH = iter(valH_loader)
        best_score = 0.0


    #models
    model_Unet = UNet3D(n_classes=CONFIG.DATASET.N_CLASSES)
    # model_Unet = UNet3D_shallow(n_classes=CONFIG.DATASET.N_CLASSES)
    # model_Unet = VNet()
    # model_Unet = NestedUNet()
    model_PAMR = PAMR()
    model_G = Generator3(input_channel=1)
    model_D = Discriminator()
    model_Blur = GaussianBlur()
    model_Lap = Laplace()
    # Optimizer
    optimizer_D = torch.optim.RMSprop(
        params=model_D.parameters(),
        lr=CONFIG.SOLVER.OPTIMIZER.LR,
        weight_decay=CONFIG.SOLVER.OPTIMIZER.WEIGHT_DECAY,
        momentum=0,#CONFIG.SOLVER.OPTIMIZER.MOMENTUM,
    )
    optimizer_U = torch.optim.Adam(
        params=model_Unet.parameters(),
        lr=CONFIG.SOLVER.OPTIMIZER.LR*10,
        weight_decay=CONFIG.SOLVER.OPTIMIZER.WEIGHT_DECAY,
        # momentum=CONFIG.SOLVER.OPTIMIZER.MOMENTUM,
    )
    optimizer_G = torch.optim.Adam(
        params=model_G.parameters(),
        lr=CONFIG.SOLVER.OPTIMIZER.LR,
        # momentum=CONFIG.SOLVER.OPTIMIZER.MOMENTUM,
    )
    # initialization
    if len(CONFIG.INIT_MODEL.UNET)>=1:
        state_dict = torch.load(CONFIG.INIT_MODEL.UNET,map_location=torch.device('cpu'))
        print("    Init:", CONFIG.INIT_MODEL.UNET)
        predict_dict = model_Unet.state_dict()
        predict_dict = {k: v for k, v in predict_dict.items() if k in state_dict}
        # state_dict.update(predict_dict)
        model_Unet.load_state_dict(state_dict, strict=False)
    if len(CONFIG.INIT_MODEL.G)>=1:
        state_dict = torch.load(CONFIG.INIT_MODEL.G)
        print("    Init:", CONFIG.INIT_MODEL.G)
        predict_dict = model_G.state_dict()
        predict_dict = {k: v for k, v in predict_dict.items() if k in state_dict}
        # state_dict.update(predict_dict)
        model_G.load_state_dict(state_dict, strict=False)
    if len(CONFIG.INIT_MODEL.D)>=1:
        state_dict = torch.load(CONFIG.INIT_MODEL.D)
        print("    Init:", CONFIG.INIT_MODEL.D)
        predict_dict = model_D.state_dict()
        predict_dict = {k: v for k, v in predict_dict.items() if k in state_dict}
        # state_dict.update(predict_dict)
        model_D.load_state_dict(state_dict, strict=False)
    model_D.to(device)
    model_G.to(device)
    model_Unet.to(device)
    model_Blur.to(device)
    model_Lap.to(device)
    model_PAMR.to(device)
    # loss
    loss_Uw = myLoss3()
    loss_D = myLossD()
    loss_U_per = nn.BCELoss()
    loss_U = myLoss2()
    loss_R = nn.MSELoss()
    # loss_R = nn.MSELoss(reduce = False)
    # loss_R = nn.L1Loss()

    scheduler_U = get_scheduler(optimizer=optimizer_U, scheduler_dict=CONFIG.SOLVER.LR_SCHEDULER_U)
    scheduler_G = get_scheduler(optimizer=optimizer_G, scheduler_dict=CONFIG.SOLVER.LR_SCHEDULER_G)
    scheduler_D = get_scheduler(optimizer=optimizer_D, scheduler_dict=CONFIG.SOLVER.LR_SCHEDULER_D)
    torch.autograd.set_detect_anomaly(True)
    
    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        CONFIG.EXP.ID,
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "jpgGenarate",
        CONFIG.EXP.ID,
        str(CONFIG.SOLVER.OPTIMIZER.LR),
    )
    makedirs(save_dir)
    # import pdb; pdb.set_trace()
    loss_zong = 1
    flag = 0
    for p in model_D.parameters():
        p.requires_grad = True
    for p in model_Unet.parameters():
        p.requires_grad = True
    for p in model_G.parameters():
        p.requires_grad = True
    for p in model_Blur.parameters():
        p.requires_grad = False
    for p in model_Lap.parameters():
        p.requires_grad = False
    ## TODO:
    ## TODO:
    ## TODO:
    ## TODO:
    ## TODO:
    loss_R_weight=0
    power_time = 1
    sen_a = []
    f = open(CONFIG.EXP.ID+'.csv','w')
    spef_a = []
    loss = 0.0
    loss_Re = 0.0
    loss_seg = 0.0
    loss_G = 0.0
    loss_classify = 0.0
    for iteration in tqdm(
        range(0, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):
        model_Unet.train()
        model_G.train()
        model_D.train()
        # Clear gradients (ready to accumulate)
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        optimizer_U.zero_grad()
        if (iteration) % 1000 == 0:
            
            cout = 0
            torch.cuda.empty_cache()
            model_Unet.eval()
            model_G.eval()
            model_D.eval()
            with torch.no_grad():
                for image_ids, images,lung_masks in tqdm(
                    loader_WL, total=1, dynamic_ncols=True
                ):
                    images = images.to(device)
                    lung_masks = lung_masks.to(device)
                    mask = model_Unet(images)
                    generateImages = model_G(images)
                    fixImages = generateImages*mask+images*(1-mask)
                    # fixImages = generateImages*mask*lung_masks+images*(1-mask*lung_masks)
                    jpg_imgs = generateImages[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    gen_img = (jpg_img+1)*127.5
                    jpg_imgs = fixImages[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    fix_img = (jpg_img+1)*127.5
                    jpg_imgs = mask[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    seg_img = (jpg_img)*255
                    jpg_imgs = images[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    ori_img = (jpg_img+1)*127.5
                    jpg_img = torch.cat((torch.cat((ori_img,seg_img),dim=0),torch.cat((fix_img,gen_img),dim=0)),dim=1)
                    jpg_img = jpg_img.detach().cpu().numpy().astype('uint8')
                    cv2.imwrite(os.path.join(save_dir,'%05d.jpg'%iteration),jpg_img)
                    cout += 1
                    if cout>0:
                        torch.cuda.empty_cache()
                        break
                for _, h_images,labels,lung_masks in tqdm(
                    loader_H, total=1, dynamic_ncols=True
                ):
                    images = h_images.to(device)
                    lung_masks = lung_masks.to(device)
                    mask = model_Unet(images)
                    generateImages = model_G(images)
                    fixImages = generateImages*mask+images*(1-mask)
                    # fixImages = generateImages*mask*lung_masks+images*(1-mask*lung_masks)
                    jpg_imgs = generateImages[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    gen_img = (jpg_img+1)*127.5
                    jpg_imgs = fixImages[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    fix_img = (jpg_img+1)*127.5
                    jpg_imgs = mask[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    seg_img = (jpg_img)*255
                    jpg_imgs = images[:,0,15]
                    jpg_img = torch.cat((torch.cat((jpg_imgs[0],jpg_imgs[1]),dim=0),torch.cat((jpg_imgs[2],jpg_imgs[3]),dim=0)),dim=1)
                    ori_img = (jpg_img+1)*127.5
                    jpg_img = torch.cat((torch.cat((ori_img,seg_img),dim=0),torch.cat((fix_img,gen_img),dim=0)),dim=1)
                    jpg_img = jpg_img.detach().cpu().numpy().astype('uint8')
                    cv2.imwrite(os.path.join(save_dir,'%05d_healthy.jpg'%iteration),jpg_img)
                    cout += 1
                    if cout>0:
                        torch.cuda.empty_cache()
                        break
                for _, h_images,labels,lung_masks in tqdm(
                    loader_L, total=1, dynamic_ncols=True
                ):
                    images = h_images.to(device)
                    lung_masks = lung_masks.to(device)
                    mask = model_Unet(images)
                    generateImages = model_G(images)
                    fixImages = generateImages*mask+images*(1-mask)
                    # fixImages = generateImages*mask*lung_masks+images*(1-mask*lung_masks)
                    jpg_imgs = generateImages[:,0,15]
                    jpg_img = jpg_imgs[0]
                    gen_img = (jpg_img+1)*127.5
                    jpg_imgs = fixImages[:,0,15]
                    jpg_img = jpg_imgs[0]
                    fix_img = (jpg_img+1)*127.5
                    jpg_imgs = mask[:,0,15]
                    jpg_img = jpg_imgs[0]
                    seg_img = (jpg_img)*255
                    jpg_imgs = images[:,0,15]
                    jpg_img = jpg_imgs[0]
                    ori_img = (jpg_img+1)*127.5
                    jpg_img = torch.cat((torch.cat((ori_img,seg_img),dim=0),torch.cat((fix_img,gen_img),dim=0)),dim=1)
                    jpg_img = jpg_img.detach().cpu().numpy().astype('uint8')
                    cv2.imwrite(os.path.join(save_dir,'%05d_labeled.jpg'%iteration),jpg_img)
                    cout += 1
                    if cout>0:
                        torch.cuda.empty_cache()
                        break
            torch.save(
                model_D.state_dict(),
                os.path.join(checkpoint_dir, "checkpointD_{}.pth".format(iteration)),
            )
            torch.save(
                model_Unet.state_dict(),
                os.path.join(checkpoint_dir, "checkpointU_{}.pth".format(iteration)),
            )
            torch.save(
                model_G.state_dict(),
                os.path.join(checkpoint_dir, "checkpointG_{}.pth".format(iteration)),
            )
            model_Unet.train()
            model_G.train()
            model_D.train()

        for _ in range(CONFIG.SOLVER.ITER_SIZE-1):
            if (iteration//2)%2==0:
                for p in model_D.parameters():
                    p.requires_grad = False
                for p in model_Unet.parameters():
                    p.requires_grad = True 
                for p in model_G.parameters():
                    p.requires_grad = True
                # optimizer_U.zero_grad() 
                optimizer_G.zero_grad()
                try:
                    _, images,lung_masks = next(iter_WL)
                except:
                    iter_WL = iter(loader_WL)
                    id_s, images,lung_masks = next(iter_WL)
                loss_G = 0
                loss = 0
                images = images.to(device)
                lung_masks = lung_masks.to(device)
                mask = model_Unet(images)
                mask_detach = mask.detach()
                generateImages = model_G(input_G)
                # fixImages = generateImages*mask+images*(1-mask)
                fixImages = generateImages*mask*lung_masks+images*(1-mask*lung_masks)
                pred,_ = model_D(fixImages)
                labels = 0
                loss_G = loss_D(pred,labels)
                if iteration > 10000:
                    if 'MIL' in CONFIG.EXP.ID:
                        max_mask = torch.max((mask*lung_masks).view(mask.size(0),-1),1).values
                        labels = torch.tensor(np.ones(max_mask.shape)).to(device).float()
                        loss_MIL = loss_D(max_mask,labels)
                        loss += loss_MIL*1000
                if 'genImgToD' in CONFIG.EXP.ID:
                    pred,_ = model_D(generateImages)
                    labels = 0
                    loss_G += loss_D(pred,labels)
                loss = loss_G
                if 'recons' in CONFIG.EXP.ID:
                    try:
                        _, h_images,labels,lung_mask = next(iter_H)
                    except:
                        iter_H = iter(loader_H)
                        _, h_images,labels,lung_mask = next(iter_H)
                    h_images = h_images.to(device)
                    input_G = h_images
                    generateImages = model_G(input_G)
                    loss_recons = ((generateImages-h_images).abs()).mean()
                    loss += loss_recons*100
                if iteration > 10000:
                    if 'perception' in CONFIG.EXP.ID:
                        if lung_mask.shape[0] != mask_detach.shape[0]:
                            print('shape different')
                        else:
                            lung_mask = lung_mask.to(device)
                            mask_detach = mask_detach*lung_mask
                            mask_detach = mask_detach.detach()
                            
                            images_addill = h_images*(1-mask_detach)+images*mask_detach
                            mask_detach = (mask_detach>0.2).float()+(mask_detach>0.8).float()
                            mask_detach[mask_detach==1]=3
                            mask_detach[mask_detach==2]=1
                            mask_detach[mask_detach==3]=2

                            logits = model_Unet(images_addill.detach())
                            loss_weak = loss_U(logits[:,0],mask_detach[:,0])*100
                            loss += loss_weak
                if loss<0.16:
                    loss = loss/1000
                loss.backward()
                optimizer_G.step()                
                optimizer_U.step()
                optimizer_U.zero_grad()
                optimizer_G.zero_grad()

                ## Segmentation loss
                try:
                    image_ids, images, labels,_ = next(iter_H)
                except:
                    iter_H = iter(loader_H)
                    image_ids, images, labels, _ = next(iter_H)
                # Propagate forward
                images = images.to(device)
                logits = model_Unet(images)
                loss_seg = loss_U(logits[:,0],labels.to(device).float())
                try:
                    ids, images, labels, _ = next(iter_L)
                except:
                    iter_L = iter(loader_L)
                    ids, images, labels, _ = next(iter_L)
                # Propagate forward
                images = images.to(device)
                logits = model_Unet(images)
                labels = labels.to(device).float()
                
                loss_L = 5*loss_Uw(logits[:,0],labels)
                loss_seg += loss_L
                loss_seg *=100
                if loss_seg<0.064:
                    loss_seg = loss_seg/100
                loss_seg.backward()                
                optimizer_U.step()
                optimizer_U.zero_grad()

                if (iteration) % 100==0:
                    print('分割器loss：',loss_seg.item())
                    print('生成器loss：',loss.item())
                    
            else:
                for p in model_Unet.parameters():
                    p.requires_grad = False
                for p in model_G.parameters():
                    p.requires_grad = False
                for p in model_D.parameters():  # reset requires_grad
                    p.requires_grad = True
                for _ in range(5):
                    loss_classify = 0
                    optimizer_D.zero_grad()
                    try:
                        _, images, lung_masks = next(iter_WL)
                    except:
                        iter_WL = iter(loader_WL)
                        _, images, lung_masks = next(iter_WL)
                    images = images.to(device)
                    mask = model_Unet(images)
                    lung_masks = lung_masks.to(device)
                    generateImages = model_G(images)
                    # fixImages = generateImages*mask+images*(1-mask)
                    fixImages = generateImages*mask*lung_masks+images*(1-mask*lung_masks)
                    pred,_ = model_D(fixImages)
                    labels = 1
                    loss_classify += loss_D(pred,labels)
                    if 'genImgToD' in CONFIG.EXP.ID:
                        pred,_ = model_D(generateImages)
                        labels = 1
                        loss_classify += loss_D(pred,labels)*((CONFIG.SOLVER.ITER_MAX-iteration)/CONFIG.SOLVER.ITER_MAX)
                    if 'origImgToD' in CONFIG.EXP.ID:
                        pred,att_layer = model_D(images)
                        loss_orig = loss_D(pred,labels)
                        if loss_orig<0.1:
                            loss_orig = loss_orig*0.0001
                        loss_classify += loss_orig
                    try:
                        _, h_images,labels,_ = next(iter_H)
                    except:
                        iter_H = iter(loader_H)
                        _, h_images,labels,_ = next(iter_H)
                    h_images = h_images.to(device)

                    pred,_ = model_D(h_images)
                    labels = 0
                    loss_classify += loss_D(pred,labels)
                    if loss_classify<0.16:
                        loss_classify = loss_classify/1000000
                    loss_classify.backward()
                    optimizer_D.step()
                if (iteration+1) % 10==0:
                    print('判别器loss：',loss_classify.item())
        # Update learning rate
        scheduler_D.step(epoch=iteration)
        scheduler_U.step(epoch=iteration)
        scheduler_G.step(epoch=iteration)
    f.close()
@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "-s",
    "--save-vis",
    is_flag=True,
    help="Save the visualization results without CRF [default: --not-save-vis")
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
def test(config_path, model_path, save_vis, cuda):
    """
    Evaluation on validation set
    """
    
    # Configuration
    CONFIG = Dict(yaml.load(config_path, Loader=yaml.FullLoader))
    print("Exp: ", CONFIG.EXP.ID)

    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # dataset_L
    dataset = myDataset(
        root = CONFIG.DATASET.ROOT,
        split = CONFIG.DATASET.SPLIT.VALID ,
        ignore_label = CONFIG.DATASET.IGNORE_LABEL,
        augment = False,
        # shuffle = False,
        crop_size = CONFIG.IMAGE.SIZE.TEST,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,shuffle=True,
    )
    model_Unet = UNet3D(n_classes=CONFIG.DATASET.N_CLASSES)
    # model_Unet = NestedUNet()
    # model_Unet = VNet()
    model_G = Generator3(input_channel=1)
    model_Blur = GaussianBlur()
    model_PAMR = PAMR()
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model_Unet.load_state_dict(state_dict)
    model_G.eval()
    model_Unet.eval()
    model_Blur.eval()
    model_Blur.to(device)
    model_Unet.to(device)
    model_G.to(device)
    model_PAMR.to(device)
    predlist,gts,ids = [],[],[]
    for image_ids, images,labels,lung_mask in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        images = images.to(device)
        probs = model_Unet(images)
        preds = probs>0.5
        ids.append(image_ids)
        predlist += list((preds.float().cpu()*lung_mask).numpy().astype('int8'))
        gts += list(labels.numpy()>0)
    score = score_p(gts,predlist,n_class=2,ids=ids)
    print('final score:',score)
    score = scores(gts,predlist,n_class=2)
    print('Class IoU: ',score['Class IoU'])

if __name__ == "__main__":
    main()

    

    



