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
import time
import torch

#TODO: Fix the process one file
def process_one_file(config, file_ID, out_name):
    print('Processing ', file_ID)
    
    # to_crop: path to image/label in org. size
    # roi_label: path to predicted label used to define ROI
    # crop_output: path to save the cropped image/label
    
    roi_image = os.path.join(out_name, os.path.splitext(os.path.basename(file_name))[0][:-4] + '.nii')
    roi_label = os.path.join(os.path.split(out_name)[0]+"/lowres_label/", os.path.splitext(os.path.basename(file_name))[0][:-4] + '_predicted_label.nii')
    hr = heartroi.HeartROI(config)       
    hr.crop_img_isotropic(to_crop, roi_label, crop_output)
    
    #TODO: include the possibility of cropping label_maps and creating distancefields (in case of creating training data)
    

def process_file_list(config, file_name, out_name):
    print('Processing filelist ', file_name)
    base_input_path = os.path.join(os.path.split(file_name)[0],os.path.split(out_name)[1])
    if os.path.split(out_name)[1] == 'img':
        suffix = ".nii"
    elif os.path.split(out_name)[1] == 'lab':
        suffix = ".nii"
    elif os.path.split(out_name)[1] == 'dfield':
        suffix = ".nii"
    else: 
        print("ERROR IN SUFFIX IN CROPPING!!")
        exit()

    names = []
    with open(file_name) as f:
        for line in f:
            line = (line.strip("/n")).strip("\n")
            if len(line) == 4:
                names.append(line)
    print('Processing ', len(names), ' meshes')

    hr = heartroi.HeartROI(config)
    for file_id in names:
        print('Processing ', file_id)
        input_file_path = os.path.join(base_input_path,file_id+suffix)
        roi_label_path = os.path.join(os.path.split(out_name)[0]+"/lowres_label/",file_id+".nii")
        roi_output_path = os.path.join(out_name, file_id+suffix)
        start_time = time.time()
        hr.crop_img_isotropic(input_file_path, roi_label_path, roi_output_path)


"""
def process_files_in_dir(config, dir_name):
    print('Processing files in  ', dir_name)
    names = Utils3D.get_mesh_files_in_dir(dir_name)
    print('Processing ', len(names), ' meshes')
    dm = deepmvlm.DeepMVLM(config)
    for file_name in names:
        print('Processing ', file_name)
        name_lm_txt = os.path.splitext(file_name)[0] + '_landmarks.txt'
        landmarks = dm.predict_one_file(file_name)
        dm.write_landmarks_as_text(landmarks, name_lm_txt)
"""


def main(config):
    name = str(config.name)
    out_name = str(config.out_name)

    if not os.path.exists(out_name):
        os.mkdir(out_name)

    if name.lower().endswith(('.nii')) and os.path.isfile(name):
        process_one_file(config, name, out_name)
    elif name.lower().endswith(('.nii.gz')) and os.path.isfile(name):
        process_one_file(config, name, out_name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        process_file_list(config, name, out_name)
    # elif os.path.isdir(name):
    #    process_files_in_dir(config, name)
    else:
        print('Cannot process (not a volume file, a filelist (.txt) or a directory)', name)


# Typical commandlines:
# --c configs/heart-basic_comp-pcrapa.json --n D:\data\IMM\Kristine\RasmusTest\ROIdetection\img\annotated_roi_images.txt -o D:\data\IMM\Kristine\RasmusTest\SDFprediction\pwrmodel_v10
# --c configs/heart-basic_comp-pcrapa.json --n D:\data\IMM\Kristine\RasmusTest\ROIdetection\img\all_roi_images.txt -o D:\data\IMM\Kristine\RasmusTest\SDFprediction\pwrmodel_v7_full

# To crop input image:         
#--c configs/config_predict.json --n C:/Users/kajul/Documents/Data/MMWHS/ct_test/ct_test_2001_image.nii.gz -o C:/Users/kajul/Documents/Data/MMWHS/ROI/
# To crop labels (training data)        :
#--c configs/config_predict.json --n C:/Users/kajul/Documents/Data/MMWHS/ct_train/MMWHS_testlist.txt --o  C:/Users/kajul/Documents/Data/MMWHS/ROI/label/       

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Deep-MVLM')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')
    args.add_argument('-o', '--oname', default=None, type=str,
                      help='name of output directory to put result files')

    global_config = ConfigParser(args)
    main(global_config)