# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:48:23 2020

@author: kajul
"""

import SimpleITK as sitk
#import Utils as Utils
# import glob
#import ROIUnet_GPU_SD3_005 as ROIUnet_GPU
import os
import sys
import subprocess
import numpy as np
#import tensorflow as tf
from skimage.morphology import label, remove_small_objects
from skimage.measure import regionprops
from numpy import loadtxt
import vtk


class HeartROI:
    """
    Extraction of ROIs from CT cardiac scans
    """
    def __init__(self,config):
        self.roi_physical_size = config["ROI_specs"]["roi_physical_size"]
        self.roi_physical_nvoxel = config["ROI_specs"]["roi_physical_nvoxel"]
        self.default_ct_value = config["ROI_specs"]["default_ct_value"]
        
        
        vtk_out = vtk.vtkOutputWindow()
        vtk_out.SetInstance(vtk_out)
        
    def crop_img_isotropic(self, file_name, roi_label_path, roi_image_path):
        """
        file_name:      Full path to .nii file to crop
        roi_label_path: Full path to the predicted labelmap for ROI
        roi_image_path: Full path to save cropped image
        """
        
        roi_physical_spacing = self.roi_physical_size / self.roi_physical_nvoxel
        print('Physical ROI spacing', roi_physical_spacing)
        
        image_label_org = sitk.Cast(sitk.ReadImage(roi_label_path), sitk.sitkFloat32)
        label_arr = sitk.GetArrayFromImage(image_label_org)
        
        properties = regionprops(label_arr.astype(int))
        center_of_mass = properties[0].centroid
        print('Center of mass', center_of_mass)
        cm_i = int(center_of_mass[2])  # Due to the z, y, x convention of numpy
        cm_j = int(center_of_mass[1])
        cm_k = int(center_of_mass[0])

        idx = (cm_i, cm_j, cm_k)
        cm_physical = image_label_org.TransformIndexToPhysicalPoint(idx)
        print('Center of mass in physical coords', cm_physical)
        
        img = sitk.Cast(sitk.ReadImage(file_name), sitk.sitkFloat32)
        #img_direction = img.GetDirection()
        img_direction = image_label_org.GetDirection()
        
        # Use the direction matrix
        # This works!
        new_origin_x = cm_physical[0] - img_direction[0] * self.roi_physical_size / 2
        new_origin_y = cm_physical[1] - img_direction[4] * self.roi_physical_size / 2
        new_origin_z = cm_physical[2] - img_direction[8] * self.roi_physical_size / 2

        roi_nvoxel = [self.roi_physical_nvoxel, self.roi_physical_nvoxel, self.roi_physical_nvoxel]

        roi_image = sitk.Image(roi_nvoxel, img.GetPixelIDValue())
        roi_image.SetOrigin([new_origin_x, new_origin_y, new_origin_z])
        roi_image.SetSpacing([roi_physical_spacing, roi_physical_spacing, roi_physical_spacing])
        roi_image.SetDirection(img_direction)

        # Make translation with no offset, since sitk.Resample needs this arg.
        translation = sitk.TranslationTransform(3)
        translation.SetOffset((0, 0, 0))
        
        interpolator = sitk.sitkLinear
        if "lab" in file_name:
            interpolator = sitk.sitkNearestNeighbor
            self.default_ct_value = 0
        if "dfield" in file_name: 
            self.default_ct_value = 1000
        
        resampled_image = sitk.Resample(img, roi_image, translation, interpolator, self.default_ct_value)
        sitk.WriteImage(resampled_image, roi_image_path)
        
        return 
        
        
        
        
        
        
        



