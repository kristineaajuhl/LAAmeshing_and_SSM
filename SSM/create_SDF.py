# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 08:59:21 2021

@author: kajul
"""

import argparse
import os
import Methods
from numpy import loadtxt

def get_all_file_names_from_txt(data_file):
    file_ids = loadtxt(data_file, dtype=str, ndmin=1, comments="#", delimiter=",", unpack=False)
    print('Read ', len(file_ids), ' file ids')
    return file_ids

def main(args):       
    filename = args.name
    print("Processing filelist: ", filename)
    name_list = get_all_file_names_from_txt(filename)
    
    mrf_exe = 'C:/Program Files/MRFTools/MRFSurface.exe'
    print("Check MRFSurface installation path at line 20")
    print(mrf_exe)
    
    base_path = os.path.split(args.surfacepath)[0] + "/LAA_decoupling/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    
    target_surfaces_dir = args.surfacepath
    if args.part == "Full":
        target_sdf_dir = base_path + "/distance_fields/"
        name_list = get_all_file_names_from_txt(filename)
        if not os.path.exists(target_sdf_dir):
            os.makedirs(target_sdf_dir)
    else: 
        print("NOT YET IMPLEMENTED!")
        
            
    N = len(name_list)
    args = list(zip(name_list, [target_surfaces_dir] * N, [target_sdf_dir] * N, [mrf_exe] * N))
    Methods.imap_unordered_bar(Methods.DistanceField, args, 12)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create distance fields from surface')
    parser.add_argument('-n', '--name', default=None, type=str,
                      help='path to filelist (.txt) to be processed')
    parser.add_argument('-s', '--surfacepath', default=None, type=str,
                        help='Path to folder with surfaces to be processed (Do not end path with /')
    parser.add_argument('-p', '--part', default=None, type=str,
                      help='Type "Full" for full LA and "LAA" for LAA only')

    args = parser.parse_args()
    main(args)