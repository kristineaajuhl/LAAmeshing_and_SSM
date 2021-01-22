# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:46:10 2021

@author: kajul
"""
from cut_class import CutClass
import argparse
from numpy import loadtxt
import os

def get_all_file_names_from_txt(data_file):
    file_ids = loadtxt(data_file, dtype=str, ndmin=1, comments="#", delimiter=",", unpack=False)
    print('Read ', len(file_ids), ' file ids')
    return file_ids

def main(args):       
    filename = args.name
    print("Processing filelist: ", filename)
    name_list = get_all_file_names_from_txt(filename)
    out_base= os.path.split(args.surfacepath)[0] + "/LAA_decoupling/"
    if not os.path.exists(out_base):
        os.mkdir(out_base)
    if not os.path.exists(out_base + "/LAA_only/"):
        os.mkdir(out_base + "/LAA_only/")
    if not os.path.exists(out_base + "/LAA_LM/"):
        os.mkdir(out_base + "/LAA_LM/")
    
    print(args.visualize)
    
    for file_id in name_list:
        print("Processing: ", file_id)
        
        predicted_surface_path = args.surfacepath + "/" + file_id + ".vtk"
        surface_in_correspondance_path = args.correspondencepath + "/" + file_id + ".vtk"
        path_ostiumIDs = os.getcwd() + "/template/ostium_ids_template.npy"
        output_path = out_base + "/LAA_only/" + file_id + ".vtk"
        LM_path = out_base + "/LAA_LM/" + file_id + ".txt"

        # Initiate class: 
        cc = CutClass(predicted_surface_path,surface_in_correspondance_path,path_ostiumIDs,output_path)
        
        #Fit plane to ostium points: 
        normal, centroid = cc.fit_plane_to_points()
        normal = cc.check_for_flipped_normal(normal)
        
        #Small brute-force pertubations
        normal_opt, centroid_opt = cc.brute_force_pertubation(normal, centroid, no=100)
        normal_opt = cc.check_for_flipped_normal(normal_opt)

        #Find the optimal cut
        cut_points = cc.find_final_cut(normal_opt,centroid_opt)
        
        #Show cut points
        if args.visualize: 
            cc.visualize_cut_points(cut_points)
        
        #Cut surface
        LAA_surface = cc.cut_example(normal_opt, centroid_opt, cut_points, thres_value = 5, vis_split = False)
        
        #Find landmarks on ostium
        cc.find_and_save_LM(LAA_surface, LM_path)
    
        #Show cut_surface
        if args.visualize: 
            cc.visualize_crop(LAA_surface)
            #cc.visualize_crop_wLM(LAA_surface)
            #TODO implement visualization of LAA with landmarks in cut_class
        
        #Save LAA
        cc.save_crop(LAA_surface)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create distance fields from surface')
    parser.add_argument('-n', '--name', default=None, type=str,
                      help='path to filelist (.txt) to be processed')
    parser.add_argument('-s', '--surfacepath', default=None, type=str,
                        help='Path to folder with surfaces to be processed (Do not end path with /) original/non-registered surfaces')
    parser.add_argument('-c','--correspondencepath', default=None, type=str,
                        help='Path to folder with surfaces in correspondence')
    parser.add_argument('-v', '--visualize',default=False, type=bool,
                        help='Vislualize processing steps')
    
    args = parser.parse_args()
    main(args)