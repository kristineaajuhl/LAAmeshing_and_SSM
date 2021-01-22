import numpy as np
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import SimpleITK
import subprocess



def RMS(sourceSurfaceFile, targetSurfaceFile,VTKobj = False):
    # Reading surfaces
    if not VTKobj:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(sourceSurfaceFile)
        reader.Update()
        sourceSurface = reader.GetOutput()
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(targetSurfaceFile)
        reader.Update()
        targetSurface = reader.GetOutput()
    else:
        sourceSurface = sourceSurfaceFile
        targetSurface = targetSurfaceFile
        
    cellLocator = vtk.vtkCellLocator()
    cellLocator.SetDataSet(targetSurface)
    cellLocator.BuildLocator()    
    sourcePoints = vtk_to_numpy(sourceSurface.GetPoints().GetData())    
    distances = np.zeros([np.size(sourcePoints, 0),1])
    idx = 0
    for point in sourcePoints:
        closestPoint = [0,0,0]
        closestPointDist2 = vtk.reference(np.float64())
        cellId = vtk.reference(1)
        subId = vtk.reference(1)
        cellLocator.FindClosestPoint(point,closestPoint,cellId,subId,closestPointDist2)
        distances[idx] = closestPointDist2
        idx += 1

    RMS = np.sqrt((distances).mean())
    return RMS

def FractionOfNegativeJacobian(jacobianFile,distanceFieldFile = None,th = 0.1):
    dirDistanceFields = 'E:/ShapeModelling/PWR/DistanceFields/'  
    Name = jacobianFile.split('\\')[-2].split('_')[0]
    if distanceFieldFile == None:
        distanceFieldFile = os.path.join(dirDistanceFields,Name,Name + '_DistanceField.mhd')
        
    jacobian = SimpleITK.ReadImage(jacobianFile)
    jacobianImage = SimpleITK.GetArrayFromImage(jacobian)
    
    distanceField = SimpleITK.ReadImage(distanceFieldFile)
    distanceFieldImage = SimpleITK.GetArrayFromImage(distanceField)
    
    jacobianImageMasked = jacobianImage[(distanceFieldImage > -th) & (distanceFieldImage < th)]
    jacobianImageNegativValues = jacobianImageMasked[jacobianImageMasked<0]
    pctNegativ = jacobianImageNegativValues.shape[0]/jacobianImageMasked.shape[0]*100
    return pctNegativ