# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:37:02 2020

@author: kajul
"""

import subprocess
import argparse
from parse_config import ConfigParser
import os
import vtk
import SimpleITK as sitk
from Remeshing import MRFremeshing

def convert_label_map_to_surface(label_name, output_file):
    print('Converting', label_name)

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(label_name)
    reader.Update()

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputConnection(reader.GetOutputPort())
    mc.SetNumberOfContours(1)
    mc.SetValue(0, 1)
    mc.Update()

    # Hacky way to get real size of image (for some reason the vtk image is incorrect)
    image_itk = sitk.Cast(sitk.ReadImage(label_name), sitk.sitkFloat32)

    origin = image_itk.GetOrigin()
    size = image_itk.GetSize()
    spacing = image_itk.GetSpacing()

    # Transform surface to fit in coordinate system
    # 6/4-2020 these transformations are found by trial and error...
    rotate = vtk.vtkTransform()
    # rotate.RotateY(180)
    rotate.RotateZ(180)
    rotate.RotateX(180)
    translate = vtk.vtkTransform()

    rotate_filter = vtk.vtkTransformFilter()
    rotate_filter.SetInputConnection(mc.GetOutputPort())
    rotate_filter.SetTransform(rotate)
    rotate_filter.Update()

    # 6/4-2020 these transformations are found by trial and error...
    t_x, t_y, t_z = -origin[0], -origin[1], origin[2] + spacing[2] * size[2]
    translate.Translate(t_x, t_y, t_z)
    translate_filter = vtk.vtkTransformFilter()
    translate_filter.SetInputData(rotate_filter.GetOutput())
    translate_filter.SetTransform(translate)
    translate_filter.Update()

    norms = vtk.vtkPolyDataNormals()
    norms.SetInputConnection(translate_filter.GetOutputPort())
    norms.SetFlipNormals(True)
    norms.SetSplitting(False)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputConnection(norms.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(output_file)
    writer.Write()
    
def remesh_surface_from_labemap(input_name,mrf_exe):
    MRF = MRFremeshing(mrf_exe, input_name, '', smooth = 0, iterSmooth = 1, displayStats = True)
    MRF.remesh(triangleFactor = 0.5)
    
def process_one_file(file_name, out_name): 
    ## TODO:  Gå igennem alle process_one_file funktionerne og sikre at de spiller sammen
    print('Processing ', file_name)
    convert_label_map_to_surface(file_name, out_name)
    
def process_file_list(file_name, out_name,mrf_exe): 
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
        label_name = os.path.join(base_input_path+'/lab/',file_id+".nii")
        surface_name = os.path.join(out_name,file_id+".vtk")
        
        convert_label_map_to_surface(label_name, surface_name)
        remesh_surface_from_labemap(surface_name,mrf_exe)
    
def main(args):
    name = str(args.name)
    out_name = str(args.oname)
    mrf_exe = str(args.mrf_dir)

    if not os.path.exists(out_name):
        os.mkdir(out_name)

    if name.lower().endswith(('.nii')) and os.path.isfile(name):
        process_one_file(name, out_name)
    elif name.lower().endswith(('.nii.gz')) and os.path.isfile(name):
        process_one_file(name, out_name)
    elif name.lower().endswith('.txt') and os.path.isfile(name):
        process_file_list(name, out_name, mrf_exe)
    # elif os.path.isdir(name):
    #    process_files_in_dir(config, name)
    else:
        print('Cannot process (not a volume file, a filelist (.txt) or a directory)', name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='heart_cropdata')
    parser.add_argument('-n', '--name', default=None, type=str,
                      help='name of file, filelist (.txt) or directory to be processed')
    parser.add_argument('-o', '--oname', default=None, type=str,
                      help='name of output directory to put result files')
    parser.add_argument('-mrf', '--mrf_dir', default=None, type=str,
                      help='path to MRF_surface.exe')
    args = parser.parse_args()

    main(args)


# python label_to_mesh.py --n E:/DATA/MEDIA/MEDIA_testIDs.txt --o E:/DATA/MEDIA/surface_model    
# python label_to_mesh.py --n E:/DATA/RAPA_LAAdata/RH_batch1_testIDs.txt --o E:/DATA/RAPA_LAAdata/surface_model