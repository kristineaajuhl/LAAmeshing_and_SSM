# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:05:24 2020

@author: kajul
"""

import argparse
from parse_config import ConfigParser
import heartpwc
# from utils3d import Utils3D
import os
import heartroi
import numpy as np
import heartpwr
import time


def process_one_file(config, file_name, out_base):
    """
    file_name: Full path to .nii image to use for input (ROI)
    out_base: Path to folder to place the predictions
    """
    print('Processing ', file_name)
    fileid = os.path.split(file_name)[-1][:-4]
    
    org_img_name = os.path.split(out_base)[0] + "/img/" +  fileid + ".nii"
    roi_pred_sdf_name = out_base + "/dfield/" + fileid + ".nii"
    label_name = out_base + "/lab/" + fileid + ".nii"
    surf_name = out_base + "/surf_model/" + fileid + ".vtk"
    
    hp = heartpwr.HeartPWR(config)
    pred_sdf, roi_img = hp.predict_one_file(file_name)   
    hp.write_sdf_as_nifti(pred_sdf, roi_img, roi_pred_sdf_name)
          
    pred_sdf, img_itk = hp.resample_to_orig_resolution(org_img_name, roi_img, pred_sdf)
    #hp.write_sdf_as_nifti(pred_sdf,img_itk, pred_sdf_name)
    
    predicted_label = hp.sdf_to_label(pred_sdf)
    hp.write_label_as_nifti(predicted_label, img_itk, label_name)
    
    hp.write_iso_surface(roi_pred_sdf_name, surf_name)
    
            
def process_file_list_pwr(config, file_name, out_base):
    """
    file_name: Full path to .txt file with file_ids
    out_name: Path to folder to place the predictions
    """
    
    print('Processing filelist ', file_name)
    base_input_path = os.path.split(file_name)[0]
    names = []
    with open(file_name) as f:
        for line in f:
            line = (line.strip("/n")).strip("\n")
            if len(line) == 4:
                names.append(line)
    hp = heartpwr.HeartPWR(config)
    print('Processing ', len(names), ' images')
    
    for file_id in names:
        print('Processing ', file_id)
        roi_img_name = os.path.join(base_input_path + "/ROI/img/",file_id+".nii")
        org_img_name = os.path.join(base_input_path + "/img/", file_id + ".nii")
        roi_pred_sdf_name = os.path.join(out_base + "/dfield/", file_id + ".nii")
        label_name = os.path.join(out_base + "/lab/", file_id + ".nii")
        surf_name = os.path.join(out_base + "/surf_model/", file_id + ".vtk")
                
        pred_sdf, roi_img = hp.predict_one_file(roi_img_name)   
        hp.write_sdf_as_nifti(pred_sdf, roi_img, roi_pred_sdf_name)
              
        pred_sdf, img_itk = hp.resample_to_orig_resolution(org_img_name, roi_img, pred_sdf)
        #hp.write_sdf_as_nifti(pred_sdf,img_itk, pred_sdf_name)
        
        predicted_label = hp.sdf_to_label(pred_sdf)
        hp.write_label_as_nifti(predicted_label, img_itk, label_name)
        
        hp.write_iso_surface(roi_pred_sdf_name, surf_name)
        


def main(config):
    name = str(config.name)    
    
    if name.lower().endswith(('.nii')) and os.path.isfile(name):
        out_name = os.path.split(os.path.split(os.path.split(name)[0])[0])[0] + "/Predictions"
        if not os.path.exists(out_name):
            os.mkdir(out_name)
        if not os.path.exists(out_name + "/lab/"):
            os.mkdir(out_name + "/lab/")
        if not os.path.exists(out_name + "/dfield/"):
            os.mkdir(out_name + "/dfield/")
        if not os.path.exists(out_name + "/surf_model/"):
            os.mkdir(out_name + "/surf_model/")
        
        process_one_file(config, name, out_name)
    elif name.lower().endswith(('.nii.gz')) and os.path.isfile(name):
        out_name = os.path.split(os.path.split(os.path.split(name)[0])[0])[0] + "/Predictions"
        if not os.path.exists(out_name):
            os.mkdir(out_name)
        if not os.path.exists(out_name + "/lab/"):
            os.mkdir(out_name + "/lab/")
        if not os.path.exists(out_name + "/dfield/"):
            os.mkdir(out_name + "/dfield/")
        if not os.path.exists(out_name + "/surf_model/"):
            os.mkdir(out_name + "/surf_model/")
        
        process_one_file(config, name, out_name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):        
        out_name = os.path.split(name)[0] + "/Predictions"
        if not os.path.exists(out_name):
            os.mkdir(out_name)
        if not os.path.exists(out_name + "/lab/"):
            os.mkdir(out_name + "/lab/")
        if not os.path.exists(out_name + "/dfield/"):
            os.mkdir(out_name + "/dfield/")
        if not os.path.exists(out_name + "/surf_model/"):
            os.mkdir(out_name + "/surf_model/")
            
        process_file_list_pwr(config, name, out_name)
    else:
        print('Cannot process (not a volume file, a filelist (.txt) or a directory)', name)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='predict-3DUnet')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')

    global_config = ConfigParser(args)
    main(global_config)
    
