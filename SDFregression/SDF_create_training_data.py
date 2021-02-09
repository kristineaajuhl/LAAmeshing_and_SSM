# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:34:09 2020

@author: kajul
"""

import subprocess
import argparse
from parse_config import ConfigParser
import os

def main(args):
    predict_from_model = False
    crop_training_images = False
    crop_training_labels = False
    create_surfacemodel = False
    create_SDF = False
    crop_SDF = True
    
    
    config_roi = str(args.config)
    name = str(args.name)
    out_name = os.path.split(name)[0] + "/ROI/"
    
    if not os.path.exists(out_name):
        os.mkdir(out_name)
    
    if predict_from_model: 
        # 1. Predict ROI on image or list of images
        subprocess.call(['python', 'predictROI.py', '--c', config_roi, '--n', name])
        # creates labelmap for ROI prediction
        
    # if crop_training_images: 
    #     # 2. Crop image to ROI
    #     roi_img_out_name = os.path.join(out_name,'img')
    #     subprocess.call(['python','crop_roi.py','--c',config_roi,'--n', name, '--o', roi_img_out_name])
    
    if crop_training_labels: 
        # 3. Crop label to ROI
        roi_lab_out_name = os.path.join(out_name,'lab')
        subprocess.call(['python','crop_roi.py','--c',config_roi,'--n', name, '--o', roi_lab_out_name])
        
    if create_surfacemodel: 
        # 4. Create a mesh model from the GT segmentation (including MRF remeshing)
        surfacemodel_path = os.path.join(os.path.split(name)[0],'surf_models')
        if not os.path.exists(surfacemodel_path):
            os.mkdir(surfacemodel_path)
        subprocess.call(['python', 'label_to_mesh.py','--n', name, '--o',surfacemodel_path, '--mrf', args.mrf_exe])
        
    if create_SDF: 
        # 5. Calculate the SDF from the GT segmentation
        # Config should have a part called "sdf_specs" including directory to MRF.exe, max dist value, 
        dfield_path = os.path.join(os.path.split(name)[0],'dfield')
        subprocess.call(['python', 'create_SDF.py', '--c', config_roi, '--n', name, '--o', dfield_path])
        
    if crop_SDF: 
        # 6. Crop the SDF to ROI
        roi_sdf_out_name = os.path.join(out_name,'dfield')
        subprocess.call(['python','crop_roi.py','--c',config_roi,'--n', name, '--o', roi_sdf_out_name])

if __name__ == '__main__':
    # TODO: Add flags to turn on cropping of labels and distance fields 
    parser = argparse.ArgumentParser(description='heart_cropdata')
    parser.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    parser.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')
    args = parser.parse_args()
    global_config = ConfigParser(parser)
    args.mrf_exe = global_config["sdf_specs"]["mrf_exe"]

    main(args)
    
    
# One file: 

# On file list: 
# python crop_trainingsdata.py --c configs/config_RH_Roi.json --n E:/DATA/MEDIA/MEDIA_testIDs.txt --o E:/DATA/MEDIA/ROI
# python crop_trainingsdata.py --c configs/config_RH_ROI.json --n E:/DATA/RAPA_LAAdata/RH_batch1_train.txt --o E:/DATA/RAPA_LAAdata/ROI