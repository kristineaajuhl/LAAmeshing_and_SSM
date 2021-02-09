# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:37:53 2020

@author: kajul
"""

import subprocess
import argparse
from parse_config import ConfigParser
import os
import vtk
import SimpleITK as sitk
from shutil import copyfile

def create_signed_distance_field(config,surface_name, dfield_name):   
    print("Creating SDF for", surface_name)
    mrf_exe = config["sdf_specs"]["mrf_exe"]
    mrf_scaling = config["sdf_specs"]["mrf_scaling"]
    max_mrf_field_size = config["sdf_specs"]["max_mrf_field_size"]
    pad_voxels = config["sdf_specs"]["pad_voxels"]
    prior_type = config["sdf_specs"]["prior_type"]
    
    if config["sdf_specs"]["show_MRFoutput"] == 0:
        fnull = None
    else: 
        fnull = open(os.devnull, 'w')
    
    if not os.path.exists(os.path.split(dfield_name)[0]):
        os.mkdir(os.path.split(dfield_name)[0])
        
    subprocess.call([mrf_exe, '-i', surface_name, '-o', dfield_name, '-t', '5', '-F', '-S', str(mrf_scaling), '-s',str(max_mrf_field_size), '-P', str(pad_voxels), '-p', str(prior_type)], stdout=fnull, stderr=fnull)

def copy_to_right_folder(dfield_name): 
    src = dfield_name + "_DistanceField.nii"
    dst = os.path.join(os.path.split(os.path.split(dfield_name)[0])[0], os.path.split(dfield_name)[1]+".nii")
    
    copyfile(src,dst)
    
def process_one_file(config, file_name, out_name): 
    ## TODO:  Gå igennem alle process_one_file funktionerne og sikre at de spiller sammen
    print('Processing ', file_name)
    create_signed_distance_field(config, file_name, out_name)

def process_file_list(config, file_name, out_name): 
    print('Processing filelist ', file_name)
    base_input_path = os.path.join(os.path.split(file_name)[0])
    
    names = []
    with open(file_name) as f:
        for line in f:
            line = (line.strip("/n")).strip("\n")
            if len(line) == 4:
                names.append(line)
    print('Processing ', len(names), ' meshes')
    
    for file_id in names:
        print('Processing ', file_id)
       
        surface_name = os.path.join(base_input_path+"/surf_models/remeshed/",file_id+".vtk")
        dfield_name = os.path.join(out_name+"/"+file_id,file_id)
        
        create_signed_distance_field(config, surface_name, dfield_name)
        copy_to_right_folder(dfield_name)
    
def main(global_config):
    name = str(global_config.name)
    out_name = str(global_config.out_name)
    config = global_config.config

    if not os.path.exists(out_name):
        os.mkdir(out_name)

    if name.lower().endswith(('.vtk')) and os.path.isfile(name):
        process_one_file(config, name, out_name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        process_file_list(config, name, out_name)
    # elif os.path.isdir(name):
    #    process_files_in_dir(config, name)
    else:
        print('Cannot process (not a volume file, a filelist (.txt) or a directory)', name)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='heart_cropdata')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')
    args.add_argument('-o', '--oname', default=None, type=str,
                      help='name of output directory to put result files')

    global_config = ConfigParser(args)
    main(global_config)


# python create_SDF.py --c configs/config_RH_PWC.json --n E:/DATA/MEDIA/MEDIA_lolIDs.txt --o E:/DATA/MEDIA/dfield/