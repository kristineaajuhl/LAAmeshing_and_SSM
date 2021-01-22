# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:02:27 2021

@author: kajul
"""

import argparse
from parse_config import ConfigParser
import heartpwc
# from utils3d import Utils3D
import os
import heartroi

def process_one_file(config, file_name, out_base):
    """
    file_name: Full path to .nii image to use for input
    out_base: Path to folder to place the ROI images
    """
    print('Processing ', file_name)
    lab_name = out_base + "/lowres_label/" + os.path.split(file_name)[1]
    roi_img_name = out_base + "/img/" + os.path.split(file_name)[1]
    
    # Predict low-resolution labelmap
    hp = heartpwc.HeartPWC(config)
    probability_map, img_itk = hp.predict_one_file(file_name)
    predicted_label = hp.probability_to_label(probability_map)
    predicted_label = hp.remove_small_blobs(predicted_label)
    hp.write_label_as_nifti(predicted_label, img_itk, lab_name)
    
    # Crop image according to predicted lowres labelmap
    hr = heartroi.HeartROI(config)       
    hr.crop_img_isotropic(file_name, lab_name, roi_img_name)
    
    
def process_file_list(config, file_name, out_base):
    """
    file_name: Full path to .txt file with file_ids
    out_base: Path to folder to place the ROI images
    """
    
    print('Processing filelist ', file_name)
    base_input_path = os.path.split(file_name)[0]+"/img/"
    names = []
    with open(file_name) as f:
        for line in f:
            line = (line.strip("/n")).strip("\n")
            if len(line) == 4:
                names.append(line)
    print('Processing ', len(names), ' images')
    
    hp = heartpwc.HeartPWC(config)
    hr = heartroi.HeartROI(config)
    
    for file_id in names:
        print('Processing ', file_id)
        full_img_name = os.path.join(base_input_path,file_id+".nii")
        lab_name = os.path.join(out_base + "/lowres_label/", file_id + '.nii')
        roi_img_name = os.path.join(out_base + "/img/", file_id + '.nii')
        
        # Predict low-resolution labelmap
        probability_map, img_itk = hp.predict_one_file(full_img_name)
        predicted_label = hp.probability_to_label(probability_map)
        predicted_label = hp.remove_small_blobs(predicted_label)
        hp.write_label_as_nifti(predicted_label, img_itk, lab_name)    

        # Crop image according to predicted lowres labelmap
        hr.crop_img_isotropic(full_img_name, lab_name, roi_img_name)

def main(config):
    name = str(config.name)
    out_name = os.path.split(os.path.split(name)[0])[0] + "/ROI/"
    
    # Create directories for outputs:
    if not os.path.exists(out_name):
        os.mkdir(out_name)
    if not os.path.exists(out_name + "/lowres_label/"):
        os.mkdir(out_name + "/lowres_label/")    
    if not os.path.exists(out_name + "/img/"):
        os.mkdir(out_name + "/img/")

    if name.lower().endswith(('.nii')) and os.path.isfile(name):
        process_one_file(config, name, out_name)
    elif name.lower().endswith(('.nii.gz')) and os.path.isfile(name):
        process_one_file(config, name, out_name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        process_file_list(config, name, out_name)
    else:
        print('Cannot process (not a volume file (.nii/.nii.gz) or a filelist (.txt))', name)


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