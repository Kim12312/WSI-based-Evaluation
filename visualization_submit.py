# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
from PIL import Image
import h5py
import openslide
from torchvision import transforms
import cv2
import math
import matplotlib.pyplot as plt

def getnumpydata(h5_file_path,slide_file_path,att_file_path,ori_file_path):
    with h5py.File(h5_file_path,'r') as f:
        dset = len(f['coords'])
        patch_level = f['coords'].attrs['patch_level']
        patch_size = f['coords'].attrs['patch_size']
    
    downsample1=2
    att_weight=np.load(att_file_path)
    att_weight1=(att_weight-min(att_weight[0]))/(max(att_weight[0])-min(att_weight[0]))+0.2
    indices = np.argsort(att_weight1[0])[::-1][:3]
    wsi = openslide.open_slide(slide_file_path+".svs")
    top_left = (0,0)
    region_size = wsi.level_dimensions[downsample1]
    imgdata=np.zeros((region_size[1],region_size[0]))
    
    for idx in range(dset):
        with h5py.File(h5_file_path,'r') as f:
            coord = f['coords'][idx]
        
        imgdata[int((coord/wsi.level_downsamples[downsample1])[1]):int((coord/wsi.level_downsamples[downsample1])[1])+patch_size*16,
                int((coord/wsi.level_downsamples[downsample1])[0]):int((coord/wsi.level_downsamples[downsample1])[0])+patch_size*16]=att_weight1[0][idx]
        for ii in range(patch_size*16):
            print(str(idx)+"/"+str(ii)+"/"+str(patch_size*16))
            for jj in range(patch_size*16):
                if int((coord/wsi.level_downsamples[downsample1])[1])+ii<len(imgdata) and int((coord/wsi.level_downsamples[downsample1])[0])+jj<len(imgdata[0]):
                    imgdata[int((coord/wsi.level_downsamples[downsample1])[1])+ii,
                        int((coord/wsi.level_downsamples[downsample1])[0])+jj]=imgdata[int((coord/wsi.level_downsamples[downsample1])[1])+ii,
                                                                             int((coord/wsi.level_downsamples[downsample1])[0])+jj]
    
    ori_data = np.load(ori_file_path)
    imgdata1=np.zeros((len(ori_data),len(ori_data[0])))
    for i in range(len(ori_data)):
        print(str(i)+"/"+str(len(ori_data)))
        for j in range(len(ori_data[0])):
            if i<len(imgdata) and j<len(imgdata[0]) and np.sum(ori_data[i,j,:])>0:
                imgdata1[i,j]=imgdata[i,j]
    return ori_data,imgdata1

case_ids=['150937']
slide_ids=['150937A1A2A3']
for index1 in range(len(case_ids)):
    try:
        slide_id=slide_ids[index1]
        wsi_names=os.listdir("E:\\14-15_prognosis_wsi\\")
        wsi_names1=[wsi_name[:-4] for wsi_name in wsi_names]
        h5_file_path="E:\\submit_codes\\patches\\"+slide_id+'.h5'
        slide_file_path="E:\\submit_codes\\14-15_prognosis_wsi\\"+slide_id
        ori_file_path="E:\\submit_codes\\oridata\\"+slide_id+".npy"
        att_file_path="E:\\submit_codes\\attresults\\"+str(case_ids[index1])+".npy"

        ori_data,imgdata1=getnumpydata(h5_file_path,slide_file_path,att_file_path,ori_file_path)
        
        
        imgdata1=imgdata1*255
        plt.figure()
        heatmap = cv2.applyColorMap(np.uint8(imgdata1), cv2.COLORMAP_HOT)
        superimposed_img2 = heatmap*0.6+ori_data*0.4
        cv2.imwrite("E:\\submit_codes\\figures\\150937_heatmap2.png", superimposed_img2)
    except:
        print("error")
