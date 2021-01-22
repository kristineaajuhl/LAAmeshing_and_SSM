# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 10:58:52 2019

@author: kajul
"""

import os
import subprocess
import multiprocessing
import Metrics
import shutil
import numpy as np
import vtk
import SimpleITK
from tqdm import tqdm

def ICP(args):
    # This method takes one source surface files and ICP align it to a set of target surfaces.

    targetNameList = args[0]                                            # List of target names
    sourceSurfaceFile = args[1]                                         # name and full path of the source surface.          
    dirSurface = args[2]                                                # Path of the surface files.
    dirOutputSurface = args[3]                                          # Output path the aligned source surfaces. 
 
    sourceName = sourceSurfaceFile.split('\\')[-1][:-4]
    
    #sourceSurfaceFile = os.path.join(dirSurface,sourceName + '.vtk')
    targetSurfaceFile = os.path.join(dirSurface,targetNameList + '.vtk')
    filename = os.path.join(dirOutputSurface, targetNameList + '.vtk')
    
    # Reading surfaces
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(sourceSurfaceFile)
    reader.Update()
    sourceSurface = reader.GetOutput()
    
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(targetSurfaceFile)
    reader.Update()
    targetSurface = reader.GetOutput()
    if targetSurface.GetNumberOfPoints() < 1:
        print('Could not read', targetSurfaceFile)
        return

    #------------------------------------------------------------------------------------------------
    # Use COM as the only LM and translate
    
    # // Compute the center of mass
    sourceCOMfilter = vtk.vtkCenterOfMass()
    sourceCOMfilter.SetInputData(sourceSurface)
    sourceCOMfilter.SetUseScalarsAsWeights(False)
    sourceCOMfilter.Update()
    sourceCOM = sourceCOMfilter.GetCenter()

    targetCOMfilter = vtk.vtkCenterOfMass()
    targetCOMfilter.SetInputData(targetSurface)
    targetCOMfilter.SetUseScalarsAsWeights(False)
    targetCOMfilter.Update()
    targetCOM = targetCOMfilter.GetCenter()
    
    # Set points in vtk   
    sourcePoints = vtk.vtkPoints()
    sourcePoints.InsertNextPoint(sourceCOM)
    source = vtk.vtkPolyData()
    source.SetPoints(sourcePoints)
    
    targetPoints = vtk.vtkPoints()
    targetPoints.InsertNextPoint(targetCOM)
    target = vtk.vtkPolyData()
    target.SetPoints(targetPoints)

    #------------------------------------------------------------------------------------------------
    # Find landmark transformation 
    landmarkTransform = vtk.vtkLandmarkTransform()
    landmarkTransform.SetSourceLandmarks(sourcePoints)
    landmarkTransform.SetTargetLandmarks(targetPoints)
    landmarkTransform.SetModeToSimilarity()
    #===========================================================================
    # landmarkTransform.SetModeToRigidBody()
    #===========================================================================
    landmarkTransform.Update()
    #------------------------------------------------------------------------------------------------
    # Apply landmarks transformation to source surface
    TransformFilter = vtk.vtkTransformPolyDataFilter()
    TransformFilter.SetInputData(sourceSurface)
    TransformFilter.SetTransform(landmarkTransform)
    TransformFilter.Update()
    transformedSourceSurface = TransformFilter.GetOutput()
  
    #------------------------------------------------------------------------------------------------
    # Find icp transformation 
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(transformedSourceSurface)
    icp.SetTarget(targetSurface)
    icp.GetLandmarkTransform().SetModeToSimilarity()
    
    icp.Modified()
    icp.Update()
    #------------------------------------------------------------------------------------------------
    # Apply icp transformation to transformed source surface
    TransformFilter = vtk.vtkTransformPolyDataFilter()
    TransformFilter.SetInputData(transformedSourceSurface)
    TransformFilter.SetTransform(icp)
    TransformFilter.Update()
    icptransformedSourceSurface = TransformFilter.GetOutput()

    
    #===========================================================================
    # RMS_transformedSurface = Metrics.RMS(icptransformedSourceSurface,targetSurface,VTKobj=True)
    # RMS_SourceTarget = Metrics.RMS(sourceSurface,targetSurface,VTKobj=True)
    #===========================================================================
    
    #------------------------------------------------------------------------------------------------
    # Write transformed source surface to .vtk file

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(icptransformedSourceSurface)
    writer.SetFileName(filename)
    writer.Write()
          
    #===========================================================================
    # return RMS_SourceTarget,RMS_transformedSurface
    #===========================================================================


def DistanceField(args):
    # This method compute distance field based on the input surface.
    # The computed distance field are stored in the ouput path "dirOutput".
    name_list = args[0]                  # List surface names.
    dir_surface = args[1]                # Path of surfaces.
    dir_output = args[2]                 # Output path, where the distance field files are stored.
    dir_mrf = args[3]

    surface_file = os.path.join(dir_surface, name_list + '.vtk')
    output_folder = os.path.join(dir_output, name_list)

    quiet = True
    if quiet:
        output_pipe = open(os.devnull, 'w')       # Ignore text output from MRF.exe.
    else:
        output_pipe = None

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_filename = os.path.join(output_folder, name_list)

    subprocess.call([dir_mrf, '-i', surface_file, '-o', output_filename, '-t', '5', '-F', '-S', '0.25', '-M', '2'],
                    stdout=output_pipe)


def CreateSamplingMask(args):
    # This method create sampling masks from distance fields 
    nameList = args[0]
    dirDistanceFields = args[1]
    maskSize = args[2]  
    dirOutput = args[3]
    
    distanceFieldFile = os.path.join(dirDistanceFields,nameList,nameList + '_DistanceField.mhd')
    if not os.path.isfile(distanceFieldFile):
        print('\n%s'%distanceFieldFile)
        return
    distanceField = SimpleITK.ReadImage(distanceFieldFile)
    mask = SimpleITK.BinaryThreshold(distanceField,lowerThreshold = -maskSize, upperThreshold = maskSize,insideValue = 1, outsideValue = 0) 
    outputFolder = os.path.join(dirOutput,nameList)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)           
    maskFilename = os.path.join(outputFolder,'Mask.mhd')
    SimpleITK.WriteImage(mask,maskFilename)
    

def DistanceFieldRegistration(args):
    # FNULL = open(os.devnull, 'w')
    quiet = True
    if quiet:
        output_pipe = open(os.devnull, 'w')       # Ignore text output from MRF.exe.
    else:
        output_pipe = None

    targetNameList = args[0]
    sourceName = args[1]       
    dirDistanceFields = args[2]
    dirSourceDistanceFields = args[3]
    dirSurface = args[4]
    dirSourceSurface = args[5]
    dirSourceMask = args[6]
    parameterFile = args[7]
    dirOutputRegistration = args[8]
    dirOutputTransformedSurfaces = args[9]
    elastix_dir = args[10]

    elastix_exe = os.path.join(elastix_dir, 'elastix.exe')
    transformix_exe = os.path.join(elastix_dir, 'transformix.exe')

    #------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Getting files from source surface
    sourceDistanceField = os.path.join(dirSourceDistanceFields,targetNameList,targetNameList + '_DistanceField.mhd')
    sourceMask = os.path.join(dirSourceMask,targetNameList,'Mask.mhd')
    sourceSurface = os.path.join(dirSourceSurface,targetNameList + '.vtk')
    # Getting files from target surface
    targetDistanceField = os.path.join(dirDistanceFields,targetNameList,targetNameList + '_DistanceField.mhd')
    targetSurface = os.path.join(dirSurface,targetNameList + '.vtk')
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Create output folder, where the registration is going to be saved
    outputFolder = os.path.join(dirOutputRegistration,targetNameList)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
        
    #------------------------------------------------------------------------------------------------------------------------------------------------------------        
    # Call elastix with b-spline transformation
    subprocess.call([elastix_exe, '-f', sourceDistanceField,'-fMask',sourceMask,'-m',targetDistanceField,'-p',parameterFile,'-out',outputFolder],stdout=output_pipe)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------        
  
    transformationFile = os.path.join(outputFolder,'TransformParameters.0.txt')
    if not os.path.isfile(transformationFile):
        print('\nNo transformation was found for:%s' % targetNameList)                                
        return
    # Create output folder for the surface transformation 
    outputfolder = os.path.join(dirOutputTransformedSurfaces,targetNameList)
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    # Call transformix to transform moving surface to fixed surface
    subprocess.call([transformix_exe, '-def',sourceSurface,'-tp',transformationFile,'-out',outputfolder],stdout=output_pipe)
    if os.path.isfile(os.path.join(outputfolder,'SurfaceTransformation.log')):
        os.remove(os.path.join(outputfolder,'SurfaceTransformation.log')) 
    os.renames(os.path.join(outputfolder,'transformix.log'), os.path.join(outputfolder,'SurfaceTransformation.log'))     
    # Call transformix to compute the Jacobian determiant
    subprocess.call([transformix_exe, '-jac','all','-out',outputfolder,'-tp',transformationFile],stdout=output_pipe)
    
    transformedSurface = os.path.join(outputfolder,'outputpoints.vtk')
    
    if os.path.isfile(transformedSurface):
        jacobianFile = os.path.join(outputfolder,'spatialJacobian.mhd')
        RMS = Metrics.RMS(transformedSurface, targetSurface)
        RMS2 = Metrics.RMS(targetSurface,transformedSurface)
        pctNegativJacobian = Metrics.FractionOfNegativeJacobian(jacobianFile, sourceDistanceField)
        fileName = os.path.join(dirOutputTransformedSurfaces,'Stats.txt')
        
        with open(fileName,'a') as f:
            f.write('%s,%0.4f,%0.4f,%0.4f\n'%(targetNameList,RMS,RMS2,pctNegativJacobian))


def MRFCor(args):
    target_name = args[0]
    # source_name = args[1]
    dir_surface = args[2]
    dir_source_surface = args[3]
    dir_output = args[4]
    a_val = args[5]
    a_valstring = '%0.3f' % a_val
    dir_mrf_cor_exe = args[6]

    dir_dump = os.path.join(dir_output, 'dump')

    quiet = True
    if quiet:
        output_pipe = open(os.devnull, 'w')  # Ignore text output from MRF.exe.
    else:
        output_pipe = None

    try:
        if not os.path.exists(dir_dump):
            os.makedirs(dir_dump)
    except (IOError, OSError):
        print("Error while creating ", dir_dump)

    # if not os.path.exists(dir_target_file):
    #    os.makedirs(dir_target_file)

    # MRFCor is designed to take a .txt file with a list of target meshes
    # Therefore we need to write a text file in which the name of the target mesh is written
    target_name = str(target_name)  # Typecast to string to be on the safe side
    target_file_temp = os.path.join(dir_dump, target_name.split('\\')[-1] + '_to_process.txt')
    target_surface = os.path.join(dir_surface, target_name + '.vtk')

    source_surface = os.path.join(dir_source_surface, target_name, 'outputpoints.vtk')
    if os.path.isfile(source_surface):
        output_temp_dir = os.path.join(dir_dump, target_name)

        if not os.path.exists(output_temp_dir):
            os.makedirs(output_temp_dir)

        with open(target_file_temp, 'w') as f:
            f.write('%s\n' % target_surface)

        subprocess.call(
            [dir_mrf_cor_exe, '-m', source_surface, '-t', target_file_temp, '-o', output_temp_dir, '-a', '-A',
             a_valstring],
            stdout=output_pipe)
        try:
            os.remove(target_file_temp)
        except (IOError, OSError):
            print("Error while deleting file ", target_file_temp)

        old_file_name = os.path.join(output_temp_dir, 'DenseMeshes', target_name + '_DenseMesh.vtk')
        new_file_name = os.path.join(dir_output, target_name + '.vtk')

        # name = target_name
        # print("NOW I AM COMPUTING NUMBER: "+str(name))
        if not os.path.isfile(new_file_name) and os.path.isfile(old_file_name):
            os.rename(old_file_name, new_file_name)

            rms0 = Metrics.RMS(target_surface, source_surface)
            rms1 = Metrics.RMS(new_file_name, target_surface)
            rms2 = Metrics.RMS(target_surface, new_file_name)

            file_name = os.path.join(dir_output, 'Stats.txt')
            if os.path.isfile(file_name):
                # print("=============== It is a file ==============")
                with open(file_name, 'a') as f:
                    f.write('%s,%0.4f,%0.4f,%0.4f\n' % (target_name, rms0, rms1, rms2))
            else:
                # print("=============== It is NOT a file ==============")
                with open(file_name, 'w') as f:
                    f.write('%s,%0.4f,%0.4f,%0.4f\n' % (target_name, rms0, rms1, rms2))
        try:
            shutil.rmtree(output_temp_dir, ignore_errors=True)
        except (IOError, OSError):
            print("Error while deleting ", output_temp_dir)


def ComputeMeanSuface(targetNameList,dirSurface,dirOutput,sourceName,StatsFile = None,thRMS = 100):
    print("================ STATSFILE ======================")
    print(StatsFile)
    rmsMetric = np.loadtxt(StatsFile, dtype=str, delimiter = ',')
    RMS = rmsMetric[:,-1]
    nameList = rmsMetric[:,0]  
    count = 1
    skibedSurface = 0
    # loading surface into vtk group
    group = vtk.vtkMultiBlockDataGroupFilter()
    for targetName in targetNameList:
        filename = os.path.join(dirSurface,targetName + '.vtk')
        if not os.path.isfile(filename):
            print('Can not read file: %s'%filename)
            continue
        nameIdx = 0
        for name in nameList:
            # if '%d' % int(name) == targetName:
            if name == targetName:
                break
            nameIdx +=1 
        if RMS[nameIdx]>thRMS:
            print('To high RMS for surface: %s RMS = %0.2f' % (targetName,RMS[nameIdx]))
            count += 1
            skibedSurface +=1
            continue                    
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        group.AddInputConnection(reader.GetOutputPort())
        count += 1
    Procrustes = vtk.vtkProcrustesAlignmentFilter()
    Procrustes.SetInputConnection(group.GetOutputPort())
    Procrustes.GetLandmarkTransform().SetModeToRigidBody()
    #===========================================================================
    # Procrustes.GetLandmarkTransform().SetModeToSimilarity()
    #===========================================================================
    Procrustes.Update()

    polydata = reader.GetOutput()
    polydata.SetPoints(Procrustes.GetMeanPoints())
    polydata = computeNormals(polydata)    
    writer = vtk.vtkPolyDataWriter()
    filename = os.path.join(dirOutput,sourceName + '.vtk')
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


# Rasmus' variant of ComputeMeanSuface
def surface_statistics(target_names, dir_surface, dir_output, source_name, stats_file=None, rms_threshold=100):
    print('Reading', stats_file)
    rms_metric = np.loadtxt(stats_file, dtype=str, delimiter=',')
    n_rms = len(rms_metric)
    surface_group = vtk.vtkMultiBlockDataGroupFilter()

    for idx in range(n_rms):
        name = rms_metric[idx, 0]
        rms = float(rms_metric[idx, -1])
        print('Surface:', name, 'RMS:', rms)
        if rms < rms_threshold:
            filename = os.path.join(dir_surface, name + '.vtk')
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(filename)
            reader.Update()
            if reader.GetOutput().GetNumberOfPoints() < 3:
                print('Could not read', filename)
            else:
                surface_group.AddInputConnection(reader.GetOutputPort())
        else:
            print('Surface', name, 'excluded due to high RMS:', rms)

    procrustes = vtk.vtkProcrustesAlignmentFilter()
    procrustes.SetInputConnection(surface_group.GetOutputPort())
    procrustes.GetLandmarkTransform().SetModeToRigidBody()
    # ===========================================================================
    # Procrustes.GetLandmarkTransform().SetModeToSimilarity()
    # ===========================================================================
    procrustes.Update()

    polydata = reader.GetOutput()
    polydata.SetPoints(procrustes.GetMeanPoints())
    polydata = computeNormals(polydata)
    writer = vtk.vtkPolyDataWriter()
    filename = os.path.join(dir_output, source_name + '_Procrustes_mean.vtk')
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()


def computeNormals(vtkPolyData):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(vtkPolyData)
    normals.ComputeCellNormalsOff()
    normals.ComputePointNormalsOn()
    normals.Update()
    polydata = normals.GetOutput()
    normalsVTK = polydata.GetPointData().GetArray("Normals")
    vtkPolyData.GetPointData().SetNormals(normalsVTK)
    return vtkPolyData

  
def imap_unordered_bar(func, args, n_processes = 2):
    p = multiprocessing.Pool(n_processes)
    res_list = []
    with tqdm(total = len(args)) as pbar:
        for res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list