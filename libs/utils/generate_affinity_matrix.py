import os
import math
import numpy as np
import scipy.io as scio
import cv2
from tqdm import tqdm
from multiprocessing import Pool
import pdb


class affinityMatrixGenerator(object):
    def __init__(self, img_root, output_root, delta_d=15, delta_r=100, pool_size=8):
        self.img_root = img_root
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        self.delta_d = delta_d
        self.delta_r = delta_r
        self.pool_size = pool_size
    
    def get_affinity_matrix(self, img_file):
        img = cv2.imread(os.path.join(self.img_root, img_file))
        (H, W, C) = img.shape
        N = H * W
        
        # define coordinate matrix
        y_coor_matrix = np.repeat(np.reshape(np.arange(0, H, 1), (H, 1)), W, axis=1)
        x_coor_matrix = np.repeat(np.reshape(np.arange(0, W, 1), (1, W)), H, axis=0)
        
        # reshape
        img = np.reshape(img, (N, C)).astype(np.float32)
        y_coor_matrix = np.reshape(y_coor_matrix, (N, 1)).astype(np.float32)
        x_coor_matrix = np.reshape(x_coor_matrix, (N, 1)).astype(np.float32)
        
        affinity_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i, N):
                distance_diff = math.pow((y_coor_matrix[i]-y_coor_matrix[j])/self.delta_d, 2) + \
                    math.pow((x_coor_matrix[i]-x_coor_matrix[j])/self.delta_d, 2)
                color_diff = 0
                for c in range(C):
                    #pdb.set_trace()
                    color_diff += math.pow((img[i, c]-img[j, c])/self.delta_r, 2)
                affinity = math.exp(-(distance_diff+color_diff)/2)
                affinity_matrix[i, j] = affinity
                affinity_matrix[j, i] = affinity
        image_id, _ = os.path.splitext(img_file)
        npy_save_path = os.path.join(self.output_root, image_id + ".npy")
        np.save(npy_save_path, affinity_matrix)
    
    def __call__(self):
        img_files = os.listdir(self.img_root)
        print("Start processing...")
        p = Pool(self.pool_size)
        for _ in tqdm(p.imap_unordered(self.get_affinity_matrix, img_files),
                           total=len(img_files)):
            pass
        p.close()
        p.join()


if __name__ == '__main__':
    img_root = "/home1/weiqiaoqiao/Data/VOC_aug/JPEGImages"
    output_root = "/home1/weiqiaoqiao/Data/VOCAugdevkit/VOC2012/AffinityMatrix"
    generator = affinityMatrixGenerator(img_root, output_root)
    generator.get_affinity_matrix("2007_000032.jpg")
