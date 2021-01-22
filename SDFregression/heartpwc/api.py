# -*- coding: utf-8 -*-
"""
Created on Tue May 12 08:08:44 2020

@author: kajul
"""

import torch
import model.model_ROI as module_arch
# from utils3d import Utils3D
# from utils3d import Render3D
# from prediction import Predict2D
from torch.utils.model_zoo import load_url
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import vtk
from skimage.morphology import label as sk_label
from skimage.morphology import remove_small_objects

models_urls = {
    'ROI_network':
        'model_ROI.pth'
}


class HeartPWC:
    def __init__(self, config):
        self.config = config
        # Get VTK errors to console instead of stupid flashing window
        vtk_out = vtk.vtkOutputWindow()
        vtk_out.SetInstance(vtk_out)
        # self.device, self.model = self._get_device_and_load_model()
        self.logger = config.get_logger('predict')
        self.device, self.model = self._get_device_and_load_model_from_url()

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "prediction will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        if n_gpu_use > 0 and torch.cuda.is_available() \
                and (torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1] < 35):
            self.logger.warning("Warning: The GPU has lower CUDA capabilities than the required 3.5 - using CPU")
            n_gpu_use = 0
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _get_device_and_load_model_from_url(self):
        logger = self.config.get_logger('predict')

        print('Initialising model')
        model = self.config.initialize('arch', module_arch)

        print('Loading checkpoint')
        model_dir = self.config['trainer']['save_dir'] + "/trained/"
        print("model_dir: ", model_dir)
        #model_name = self.config['name']
        model_name = "ROI_network"
        print("model_name: ", model_name)
        check_point_name = models_urls[model_name]
        print("check_point_name: ", check_point_name)

        print('Getting device')
        device, device_ids = self._prepare_device(self.config['n_gpu'])

        logger.info('Loading checkpoint: {}'.format(check_point_name))

        checkpoint = load_url(check_point_name, model_dir, map_location=device)
        #checkpoint = load_url( model_dir, map_location=device)

        state_dict = checkpoint['state_dict']
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()
        return device, model


    def predict_one_file(self, file_name):
        #print("XXXXX: ", file_name)
        image_size = self.config['data_loader']['args']['image_size']
        n_classes = self.config['data_loader']['args']['n_classes']
        
        img_itk = sitk.Cast(sitk.ReadImage(file_name), sitk.sitkFloat32)
        img_in = sitk.GetArrayFromImage(img_itk)
        
        org_size = img_itk.GetSize()
                
        if org_size[0] == image_size:
            img_in = np.clip(img_in,-1000,1000)/1000
        else:
            img_in = self.resample_sitk(img_itk, image_size, interpolator = sitk.sitkLinear) 
            img_in = np.clip(img_in,-1000,1000)/1000
            

        # ---------------------- Reshape to tensor --------------------------------
        # print('Reshaping to tensors')
        # Change data type (float32 requires half the memory compared to float64)
        img_in = np.float32(img_in)

        image = torch.from_numpy(img_in.reshape(1, 1, image_size, image_size, image_size))
        image = image.to(self.device)

        # Reshuffle data
        # img = image.permute(0, 4, 1, 2, 3)  # from NHWC to NCHW
        with torch.no_grad():
            output = self.model(image)
            prob = torch.nn.functional.softmax(output,dim=1) #probaiility map

            prop_temp = (prob.cpu()).numpy() #probability map on cpu 
   
        if org_size == image_size: 
            prob_map = np.argmax(prop_temp[0,:,:,:,:], axis = 0) 
        else: 
            prob_map = np.zeros((n_classes,org_size[2],org_size[0],org_size[1]))
            for c in range(n_classes):
                label_itk = sitk.GetImageFromArray(prop_temp[0, c, :, :, :]) #ITK probability map for all classes
                prob_map[c,:,:,:] = self.resample_sitk(label_itk, org_size, interpolator = sitk.sitkLinear)
                
            label = np.argmax(prob_map,axis=0).astype(np.float64)
    
#        print("UNIQUE: ",np.unique(label))
        label_itk = sitk.GetImageFromArray(label)  
        label_itk.CopyInformation(img_itk)
        
        prob_itk = sitk.GetImageFromArray(prob_map[1,:,:,:])
        prob_itk.CopyInformation(img_itk)

        return prob_map, img_itk
    
    def resample_to_orig_resolution(self, org_img_path, roi_img, to_resample):
        """
        org_img_path: path to original image
        roi_img: input image
        to_resample: numpy array to resample to the original resolution
        
        resampled_itk: itk after resampling
        """
        if not to_resample.ndim == 4:
            to_resample = np.expand_dims(to_resample,axis=0)
        
        n_channels = to_resample.shape[0]
        img_itk = sitk.ReadImage(org_img_path)
        org_size = img_itk.GetSize()
        resampled = np.zeros((2,org_size[2],org_size[0],org_size[1]))
        
        for c in range(n_channels):
            to_resample_itk = sitk.GetImageFromArray(to_resample[c,:,:,:])
            to_resample_itk.CopyInformation(roi_img)
        
            if not img_itk.GetSize() == to_resample_itk.GetSize():
                translation = sitk.TranslationTransform(3)
                translation.SetOffset((0,0,0))
                resampled_itk = sitk.Resample(to_resample_itk, img_itk, translation, sitk.sitkLinear)
                resampled[c,:,:,:] = sitk.GetArrayFromImage(resampled_itk)
        
        return resampled, img_itk
        
    def probability_to_label(self, probability_map):
        """
        Converts probability map to label map using argmax
        """
        label_map = np.argmax(probability_map, axis=0).astype(np.float64)
        
        return label_map

    def resample_sitk(self,img,img_size,interpolator):
        """
        Helper function to resample images
        image: 
        img_size: 
        interpolator:
        """
        reference_physical_size = np.zeros(3) 
        reference_physical_size = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), 
                                    img.GetSpacing(), reference_physical_size)]
        
        # Create the reference image with a same origin and direction
        reference_origin = img.GetOrigin() #alternative: np.zeros(dim)
        reference_direction = img.GetDirection() # alternative: np.identity(dim).flatten()
        
        # Create reference image size and spacing
        if isinstance(img_size,int): #Create same number of voxels in all dim
            reference_size = [img_size]*3 
        else:
            reference_size = img_size    
        
        reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]
        
        # Create reference image
        reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)
        
        # Make translation with no offset, since sitk.Resample needs this arg.
        translation = sitk.TranslationTransform(3)
        translation.SetOffset((0, 0, 0)) 
        
        # Create final reasampled image
        resampled_image = sitk.Resample(img, reference_image, translation, interpolator)
        
        return sitk.GetArrayFromImage(resampled_image)
    
    def remove_small_blobs(self,predicted_label):
        """ Removes small blobs in the prediction
        predicted_label: Predicted labelmap as numpy array
        """

        if not np.max(predicted_label) == 1: #TODO: Check the multiclass setting later
            label = np.zeros(predicted_label.shape)
            for c in range(1,predicted_label.max+1):
                one_hot = np.zeros(predicted_label.shape)
                one_hot[predicted_label == c] = 1
                label_blob = sk_label(one_hot, neighbors = 8, background = None)
                label_one_hot = remove_small_objects(label_blob, min_size=500)
                label[label_one_hot>0] == c
            
        else:
            label_blob = sk_label(predicted_label, neighbors = 8, background = None)
            label = remove_small_objects(label_blob, min_size=500)>0
        
        return label.astype(np.float64)

    @staticmethod
    def write_label_as_nifti(label, information_itk, file_name):
        if not os.path.exists(os.path.split(file_name)[0]):
            print("Creating: ", os.path.split(file_name)[0])
            os.mkdir(os.path.split(file_name)[0]) 
        
        label_itk = sitk.GetImageFromArray(label)
        label_itk.CopyInformation(information_itk)
        
        sitk.WriteImage(label_itk, file_name)
        
    @staticmethod
    def write_probmap_as_nifti(prob_map, information_itk, file_name):
        if not os.path.exists(os.path.split(file_name)[0]):
            print("Creating: ", os.path.split(file_name)[0])
            os.mkdir(os.path.split(file_name)[0])
        if prob_map.ndim == 3:
            prob_map_itk = sitk.GetArrayFromImage(prob_map)
            prob_map_itk.CopyInformation(information_itk)
            sitk.WriteImage(prob_map_itk, file_name)
            
        else: 
            n_channels = prob_map.shape[0]
            for c in range(n_channels):
                channel_file_name = file_name[:-4]+"_channel"+str(c)+file_name[-4::]
                prob_map_itk = sitk.GetImageFromArray(prob_map[c,:,:,:])
                prob_map_itk.CopyInformation(information_itk)
                
                sitk.WriteImage(prob_map_itk, channel_file_name)
            

    @staticmethod
    def write_isosurface_from_labelmap(label_name, surface_name):
        if not os.path.exists(os.path.split(surface_name)[0]):
            print("Creating: ", os.path.split(surface_name)[0])
            os.mkdir(os.path.split(surface_name)[0])
        
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(label_name)
        reader.Update()
        
        annotation = reader.GetOutput()
            
        ## Marching cubes. 
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(annotation)
        mc.SetValue(0,0.5)
        mc.Update()
        surf = mc.GetOutput()
        
        ## Choose only largest component
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputData(surf)
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()
            
        ## Transform surface to fit in coordinate system
        transform = vtk.vtkTransform()
        transform.SetMatrix(reader.GetQFormMatrix())
        tfilter = vtk.vtkTransformPolyDataFilter()
        tfilter.SetTransform(transform)
        tfilter.SetInputData(connectivity_filter.GetOutput())
        tfilter.Update()
            
        # Save as vtk: 
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputConnection(tfilter.GetOutputPort())
        norms.SetFlipNormals(True)
        norms.SetSplitting(False)
    
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(norms.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileName(surface_name)
        writer.Write()
        
            
    @staticmethod
    def write_isosurface_from_probabilitymap(prob_map_name, surface_name):
        if not os.path.exists(os.path.split(surface_name)[0]):
            print("Creating: ", os.path.split(surface_name)[0])
            os.mkdir(os.path.split(surface_name)[0])
            
        if not os.path.exists(prob_map_name):
            prob_map_name = prob_map_name[:-4]+"_channel1"+prob_map_name[-4::]
        
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(prob_map_name)
        reader.Update()
        
        annotation = reader.GetOutput()
            
        ## Marching cubes. 
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(annotation)
        mc.SetValue(0,0.5)
        mc.Update()
        surf = mc.GetOutput()
        
        ## Choose only largest component
        connectivity_filter = vtk.vtkPolyDataConnectivityFilter()
        connectivity_filter.SetInputData(surf)
        connectivity_filter.SetExtractionModeToLargestRegion()
        connectivity_filter.Update()
            
        ## Transform surface to fit in coordinate system
        transform = vtk.vtkTransform()
        transform.SetMatrix(reader.GetQFormMatrix())
        tfilter = vtk.vtkTransformPolyDataFilter()
        tfilter.SetTransform(transform)
        tfilter.SetInputData(connectivity_filter.GetOutput())
        tfilter.Update()
            
        # Save as vtk: 
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputConnection(tfilter.GetOutputPort())
        norms.SetFlipNormals(True)
        norms.SetSplitting(False)
    
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(norms.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileName(surface_name)
        writer.Write()

    @staticmethod
    def decimate_isosurface(surf_name,out_name, factor):
        if not os.path.exists(os.path.split(out_name)[0]):
            os.mkdir(os.path.split(out_name)[0])
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(surf_name)
        reader.Update()
        
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(reader.GetOutput())
        decimate.SetTargetReduction(factor)
        decimate.PreserveTopologyOn()
        decimate.Update()
        
        writer = vtk.vtkPolyDataWriter()
        writer.SetInputConnection(decimate.GetOutputPort())
        writer.SetFileTypeToBinary()
        writer.SetFileName(out_name)
        writer.Write()

    @staticmethod
    def write_iso_surface(file_name): #TODO: Update this one
        mrf_exe = 'D:/rrplocal/bin/RAPACodeVS2019/MRFSurface/Release/MRFSurface.exe'  # TODO: Should not be set here

        basename, ext = os.path.splitext(file_name)
        surf_name = basename + '_surface.vtk'
        temp_name = basename + '_surface_temp.vtk'

        print('Extracting surface from ', file_name)

        dfield = sitk.Cast(sitk.ReadImage(file_name), sitk.sitkFloat32)
        origin = dfield.GetOrigin()
        size = dfield.GetSize()

        target_edge_length = -1
        edge_length = target_edge_length
        if target_edge_length <= 0:
            spacing = dfield.GetSpacing()
            min_space = min(spacing)
            edge_length = min_space * 0.5

        # MRFSurface is called with additional arguments
        # l : Means that only larget connected component surface is kept
        # E : Target edge length for remeshing (if set to -1 a value is automatically computed)
        # (iso-surface extraction and remeshing is done in one operation)
        subprocess.call([mrf_exe, '-i', file_name, '-o', temp_name, '-l', '-E', str(edge_length)])

        fin = vtk.vtkPolyDataReader()
        fin.SetFileName(temp_name)
        fin.Update()

        surf = fin.GetOutput()

        # Transform surface to fit in coordinate system
        # 6/4-2020 these transformations are found by trial and error...
        rotate = vtk.vtkTransform()
        # rotate.RotateY(180)
        rotate.RotateZ(180)
        rotate.RotateX(180)
        translate = vtk.vtkTransform()
        # translate.Translate(t_x, t_y, t_z)

        rotate_filter = vtk.vtkTransformFilter()
        rotate_filter.SetInputData(surf)
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
        writer.SetFileName(surf_name)
        writer.Write()

        os.remove(temp_name)