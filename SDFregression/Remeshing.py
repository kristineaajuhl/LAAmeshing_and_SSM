# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:57:37 2019

@author: kajul
"""

import vtk
from vtk.util.numpy_support import vtk_to_numpy
import subprocess
import numpy as np
import os

class MRFremeshing():
    """
    Remesh the surface using Markov Random Field reconstruction
    """
    def __init__(self,mrf_exe, surfaceFile: str,outputdir: str, smooth: bool, iterSmooth: 0, displayStats: False):
        """
        Remesh the surface using Markov Random Field reconstruction
        Args: 
            surfaceFile: Full path to surface file in .vtk format
            outputdir: Path to remeshed surface
            smooth: Boolean. Perform smoothing of surface before MRF
            iterSmooth: Optional. Iterations for smoothing
            displayStats: Boolean. Display information on surface before and after
        """
        self.surfaceFile = surfaceFile
        self.smooth = smooth
        if self.smooth:
            if self.iterSmooth == 0:
                print("Choose number of iterations larger than zero!")
                exit
        self.displayStats = displayStats
        self.outputfile = os.path.split(surfaceFile)[0]+"/remeshed/"+os.path.split(surfaceFile)[1]
        if not os.path.exists(os.path.split(surfaceFile)[0]+"/remeshed/"):
            os.mkdir(os.path.split(surfaceFile)[0]+"/remeshed/")
        #self.dirMRF = "C:/Program Files/MRFSurface/MRFSurface.exe"
        self.dirMRF = mrf_exe#config["sdf_specs"]["mrf_exe"]#"C:/Program Files/MRFTools/MRFSurface.exe"
        
        
    def remesh(self,triangleFactor = 0.5):
        """
        Args: 
            Triangle factor: (default 0.5)
        """
        # Load surface
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.surfaceFile)
        reader.Update()
        pd = reader.GetOutput()
        
        if self.displayStats: 
            PDpoints = vtk_to_numpy(pd.GetPoints().GetData())

            print("Print number of points before: ",PDpoints.shape[0])
            
            ## Triangle area before: 
            area = np.zeros(pd.GetNumberOfCells())
            for i in range(pd.GetNumberOfCells()):
                cell = pd.GetCell(i)
                p0 = cell.GetPoints().GetPoint(0)
                p1 = cell.GetPoints().GetPoint(1)
                p2 = cell.GetPoints().GetPoint(2)
                
                area[i] = vtk.vtkTriangle().TriangleArea(p0,p1,p2)
                
            print("Triangle area before: ",np.mean(area))
        
        if self.smooth:
            smoothFilter = vtk.vtkSmoothPolyDataFilter()
            smoothFilter.SetInputConnection(reader.GetOutputPort())
            smoothFilter.SetNumberOfIterations(self.iterSmooth)
            smoothFilter.SetRelaxationFactor(0.1)
            smoothFilter.FeatureEdgeSmoothingOff()
            smoothFilter.BoundarySmoothingOn()
            smoothFilter.Update()
            
            ## Compute normals
            normalGenerator = vtk.vtkPolyDataNormals()
            normalGenerator.SetInputConnection(smoothFilter.GetOutputPort())
            normalGenerator.ComputePointNormalsOn()
            normalGenerator.ComputeCellNormalsOn()
            normalGenerator.Update()
            
            smooth_outputname = self.outputfile[:-12]+"smooth.vtk"
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(normalGenerator.GetOutput())
            writer.SetFileTypeToBinary()
            writer.SetFileName(smooth_outputname)
            writer.Write()
            
            inputfilename = smooth_outputname #Input to MRF reconstruction
        else: 
#            normalGenerator = vtk.vtkPolyDataNormals()
#            normalGenerator.SetInputConnection(reader.GetOutputPort())
#            normalGenerator.ComputePointNormalsOn()
#            normalGenerator.ComputeCellNormalsOn()
#            normalGenerator.Update()
            
            inputfilename = self.surfaceFile #Input to MRF reconstruction
        
        ## Remesh
        quiet = True
        if quiet:
            output_pipe = open(os.devnull, 'w')       # Ignore text output from MRF.exe.
        else:
            output_pipe = None
        subprocess.call([self.dirMRF,'-t','2','-p','0','-x',str(triangleFactor),'-i',inputfilename,'-o',self.outputfile],stdout=output_pipe)
        
        
        if self.displayStats: 
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(self.outputfile)
            reader.Update()
            pd = reader.GetOutput()
            PDpoints = vtk_to_numpy(pd.GetPoints().GetData())

            print("Print number of points after: ",PDpoints.shape[0])
            
            ## Triangle area before: 
            area = np.zeros(pd.GetNumberOfCells())
            for i in range(pd.GetNumberOfCells()):
                cell = pd.GetCell(i)
                p0 = cell.GetPoints().GetPoint(0)
                p1 = cell.GetPoints().GetPoint(1)
                p2 = cell.GetPoints().GetPoint(2)
                
                area[i] = vtk.vtkTriangle().TriangleArea(p0,p1,p2)
                
            print("Triangle area after: ",np.mean(area))
        
