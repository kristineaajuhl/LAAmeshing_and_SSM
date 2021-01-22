# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 09:31:58 2021

@author: kajul
"""

import argparse
import os
import Methods
from numpy import loadtxt
import vtk

def get_all_file_names_from_txt(data_file):
    file_ids = loadtxt(data_file, dtype=str, ndmin=1, comments="#", delimiter=",", unpack=False)
    print('Read ', len(file_ids), ' file ids')
    return file_ids

def main(args):
    #====================     SETUP    ========================= 
    mrf_exe = 'C:/Program Files/MRFTools/MRFSurface.exe'
    mrf_cor_exe = 'C:/Program Files/MRFTools/MRFCor.exe'
    elastix_dir = 'C:/Program Files/elastix-4.9.0/'
    print("Check MRF and elastix directories and set in line 19-22")
    print(mrf_exe)
    print(mrf_cor_exe)
    print(elastix_dir)
    
    # Get VTK errors to console instead of stupid flashing window
    vtk_out = vtk.vtkOutputWindow()
    vtk_out.SetInstance(vtk_out)

    rms_thres = 2  # RMS threshold. The script only compute mean shape based on point cor less than threshold
    run_icp = True
    run_mrf_sdf = True
    run_masks = True
    run_registration = True
    run_mrf_cor = True
    n_threads = 12

    filename = args.name
    base_path = os.path.split(args.surfacepath)[0] + "/LAA_decoupling/"
    target_surfaces_dir = args.surfacepath
    target_sdf_dir = base_path + "/distance_fields/"
    mrf_cor_output_dir = base_path + "/output/"
    target_name_list = get_all_file_names_from_txt(filename)
    
    if args.template == 0 and args.part == "Full":
        source_name = "template"
        source_surface_file = os.getcwd() + "/template/full_template.vtk"
        parameter_file = os.getcwd() + "/template/SplineTransform_noLM.txt"
        n_iterations = 1
    elif args.template == 1 and args.part == "Full":
        source_name = target_name_list[0]
        source_surface_file = target_surfaces_dir + "/" + source_name + ".vtk"
        parameter_file = os.getcwd() + "/template/SplineTransform_noLM.txt"
        n_iterations = 3
    if args.template == 0 and args.part == "LAA":
        source_name = "template"
        source_surface_file = os.getcwd() + "/template/laa_template.vtk"
        parameter_file = os.getcwd() + "/template/SplineTransform_wLM.txt"
        n_iterations = 1
    elif args.template == 1 and args.part == "LAA":
        source_name = target_name_list[0]
        source_surface_file = target_surfaces_dir + "/" + source_name + ".vtk"
        parameter_file = os.getcwd() + "/template/SplineTransform_wLM.txt"
        n_iterations = 3
    
    #====================     DO THE REGISTRATION     =========================    
    N = len(target_name_list)
    for iteration in range(n_iterations):
        iteration_output_dir = os.path.join(mrf_cor_output_dir, str(iteration) + '_' + source_name)
        
        if iteration > 0:
            iteration_out_dir_old = os.path.join(mrf_cor_output_dir, str(iteration - 1) + '_' + source_name)
            source_surface_file = os.path.join(iteration_out_dir_old, source_name + '_Procrustes_mean.vtk')

        if not os.path.exists(iteration_output_dir):
            os.makedirs(iteration_output_dir)
        
        print('ICP aligning source surface to all target surface')
        icp_output_surface = os.path.join(iteration_output_dir, 'ICP_Surface')
        if not os.path.exists(icp_output_surface):
            os.makedirs(icp_output_surface)
        if run_icp:
            icpArgs = list(
                zip(target_name_list, [source_surface_file] * N, [target_surfaces_dir] * N, [icp_output_surface] * N))
            Methods.imap_unordered_bar(Methods.ICP, icpArgs, n_threads)
        
        print('Create Distance Fields from ICP aligned surfaces')
        aligned_source_sdf = os.path.join(iteration_output_dir, 'DistanceFields')
        if not os.path.exists(aligned_source_sdf):
            os.makedirs(aligned_source_sdf)
        if run_mrf_sdf:
            args = list(zip(target_name_list, [icp_output_surface] * N, [aligned_source_sdf] * N, [mrf_exe] * N))
            Methods.imap_unordered_bar(Methods.DistanceField, args, n_threads)
            
        print('Create sampling mask')
        mask_size = 6
        out_mask = os.path.join(iteration_output_dir, 'Mask', str(mask_size))
        if not os.path.exists(out_mask):
            os.makedirs(out_mask)
        if run_masks:
            args = list(zip(target_name_list, [aligned_source_sdf] * N, [mask_size] * N, [out_mask] * N))
            Methods.imap_unordered_bar(Methods.CreateSamplingMask, args, n_threads)
            
        print('Distance field based registration')
        registration_out_dir = os.path.join(iteration_output_dir, 'Registration')
        if not os.path.join(registration_out_dir):
            os.makedirs(registration_out_dir)
        transformed_surface_dir = os.path.join(iteration_output_dir, 'TransformedSurface')
        if not os.path.join(transformed_surface_dir):
            os.makedirs(transformed_surface_dir)

        print('Running distance Field registration')
        if run_registration:
            args = list(zip(
                target_name_list, [source_name] * N, [target_sdf_dir] * N, [aligned_source_sdf] * N,
                                  [target_surfaces_dir] * N,
                                  [icp_output_surface] * N, [out_mask] * N, [parameter_file] * N,
                                  [registration_out_dir] * N, [transformed_surface_dir] * N,
                                  [elastix_dir] * N))
            Methods.imap_unordered_bar(Methods.DistanceFieldRegistration, args, n_threads)

        print('Running MRF point correspondence')
        mrf_cor_out_dir = os.path.join(iteration_output_dir, 'MRFCor')
        if not os.path.exists(mrf_cor_out_dir):
            os.makedirs(mrf_cor_out_dir)
        if run_mrf_cor:
            # N = 1
            args = list(zip(target_name_list, [source_name] * N, [target_surfaces_dir] * N,
                            [transformed_surface_dir] * N,
                            [mrf_cor_out_dir] * N, [1] * N, [mrf_cor_exe] * N))
            Methods.imap_unordered_bar(Methods.MRFCor, args, n_threads)
            
        # -------------------------------------------------------------------------------------------------------------
        # Compute mean surface          
        stats_file = os.path.join(mrf_cor_out_dir, 'Stats.txt')
        Methods.surface_statistics(target_name_list, mrf_cor_out_dir, iteration_output_dir, source_name, stats_file,
                                   rms_thres)

        # Methods.ComputeMeanSuface(listTargetNames, dirOutputMRF, iterationOutputFolder,sourceName,StatsFile,RMSTH)
        # meanSurfaceFilename = os.path.join(iterationOutputFolder,sourceName + '.vtk')

        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Register surfaces to common template')
    parser.add_argument('-n', '--name', default=None, type=str,
                      help='path to filelist (.txt) to be processed')
    parser.add_argument('-p', '--part', default=None, type=str,
                      help='Type "Full" for full LA and "LAA" for LAA only')
    parser.add_argument('-s', '--surfacepath', default=None, type=str,
                        help='Path to folder with surfaces to be processed (Do not end path with /')
    parser.add_argument('-t', '--template', default=0, type=int,
                        help='Use default template (0) or create new from your data (1)')

    args = parser.parse_args()
    main(args)