import torch
import model.model_PWR as module_arch
# from utils3d import Utils3D
# from utils3d import Render3D
# from prediction import Predict2D
from torch.utils.model_zoo import load_url
import SimpleITK as sitk
import numpy as np
import os
import subprocess
import vtk

models_urls = {
    'PWR_network':
        'model_PWR.pth'
}


class HeartPWR:
    def __init__(self, config):
        self.config = config
        # Get VTK errors to console instead of stupid flashing window
        vtk_out = vtk.vtkOutputWindow()
        vtk_out.SetInstance(vtk_out)
        # self.device, self.model = self._get_device_and_load_model()
        self.logger = config.get_logger('predict')
        self.device, self.model = self._get_device_and_load_model_from_url()
        self.mrf_exe = config["sdf_specs"]["mrf_exe"]

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
        model_name = self.config['name']
        check_point_name = models_urls[model_name]

        print('Getting device')
        device, device_ids = self._prepare_device(self.config['n_gpu'])

        logger.info('Loading checkpoint: {}'.format(check_point_name))

        checkpoint = load_url(check_point_name, model_dir, map_location=device)

        state_dict = checkpoint['state_dict']
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()
        return device, model

    # Deprecated - should not be used
    def _get_device_and_load_model(self):
        logger = self.config.get_logger('test')

        print('Initialising model')
        model = self.config.initialize('arch', module_arch)
        # logger.info(model)

        print('Loading checkpoint')
        model_name = self.config['name']
        image_channels = self.config['data_loader']['args']['image_channels']
        if model_name == "MVLMModel_DTU3D":
            if image_channels == "geometry":
                check_point_name = 'saved/trained/MVLMModel_DTU3D_geometry.pth'
            elif image_channels == "RGB":
                check_point_name = 'saved/trained/MVLMModel_DTU3D_RGB_07092019.pth'
            elif image_channels == "depth":
                check_point_name = 'saved/trained/MVLMModel_DTU3D_Depth_19092019.pth'
            elif image_channels == "RGB+depth":
                check_point_name = 'saved/trained/MVLMModel_DTU3D_RGB+depth_20092019.pth'
            else:
                print('No model trained for ', model_name, ' with channels ', image_channels)
                return None, None
        elif model_name == 'MVLMModel_BU_3DFE':
            if image_channels == "RGB":
                check_point_name = 'saved/trained/MVLMModel_BU_3DFE_RGB_24092019_6epoch.pth'
            else:
                print('No model trained for ', model_name, ' with channels ', image_channels)
                return None, None
        else:
            print('No model trained for ', model_name)
            return None

        logger.info('Loading checkpoint: {}'.format(check_point_name))

        device, device_ids = self._prepare_device(self.config['n_gpu'])

        checkpoint = torch.load(check_point_name, map_location=device)

        state_dict = checkpoint['state_dict']
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()
        return device, model

    def predict_one_file(self, file_name):
        img_itk = sitk.Cast(sitk.ReadImage(file_name), sitk.sitkFloat32)

        # Obtain as arrays
        img_array = sitk.GetArrayFromImage(img_itk)

        # We do not reshape here. Just checking sizes
        image_size = img_array.shape[0]

        # Clamp to image [-1000,1000] and normalise to [-1,1]
        img_array = np.clip(img_array, -1000, 1000)
        img_array = img_array / 1000

        # ---------------------- Reshape to tensor --------------------------------
        # print('Reshaping to tensors')
        # Change data type (float32 requires half the memory compared to float64)
        img_array = np.float32(img_array)

        image = torch.from_numpy(img_array.reshape(1, 1, image_size, image_size, image_size))
        image = image.to(self.device)

        # Reshuffle data
        # img = image.permute(0, 4, 1, 2, 3)  # from NHWC to NCHW
        with torch.no_grad():
            output = self.model(image)

            out_temp = (output.cpu()).numpy()

            out_itk = sitk.GetImageFromArray(out_temp[0, 0, :, :, :])
            out_itk.SetDirection(img_itk.GetDirection())
            out_itk.SetSpacing(img_itk.GetSpacing())
            out_itk.SetOrigin(img_itk.GetOrigin())
            
        return out_temp[0,0,:,:,:], img_itk
    
    def resample_to_orig_resolution(self, org_img_path, roi_img, to_resample):
        """
        org_img_path: path to original image
        roi_img: input image
        to_resample: numpy array to resample to the original resolution
        
        resampled_itk: itk after resampling
        """
        
        img_itk = sitk.ReadImage(org_img_path)
        
        to_resample_itk = sitk.GetImageFromArray(to_resample)
        to_resample_itk.CopyInformation(roi_img)
        
        if not img_itk.GetSize() == to_resample_itk.GetSize():
            translation = sitk.TranslationTransform(3)
            translation.SetOffset((0,0,0))
            resampled_itk = sitk.Resample(to_resample_itk, img_itk, translation, sitk.sitkLinear)
            resampled = sitk.GetArrayFromImage(resampled_itk)
        else: 
            resampled = to_resample
        
        return resampled, img_itk
    
    @staticmethod
    def sdf_to_label(sdf):
        label = (sdf<0).astype(int)
        return label

    @staticmethod
    def write_sdf_as_nifti(sdf,information_itk, file_name):
        if not os.path.exists(os.path.split(file_name)[0]):
            os.mkdir(os.path.split(file_name)[0])
        
        sdf_itk = sitk.GetImageFromArray(sdf)
        sdf_itk.CopyInformation(information_itk)
        
        sitk.WriteImage(sdf_itk, file_name)
        
    @staticmethod
    def write_label_as_nifti(label, information_itk, file_name):
        if not os.path.exists(os.path.split(file_name)[0]):
            os.mkdir(os.path.split(file_name)[0])
        
        label_itk = sitk.GetImageFromArray(label)
        label_itk.CopyInformation(information_itk)
        
        sitk.WriteImage(label_itk, file_name)

    def write_iso_surface(self, sdf_name, surface_name):
        if not os.path.exists(os.path.split(surface_name)[0]):
            os.mkdir(os.path.split(surface_name)[0])
        
        #mrf_exe = 'D:/rrplocal/bin/RAPACodeVS2019/MRFSurface/Release/MRFSurface.exe'  # TODO: Should not be set here

        basename, ext = os.path.splitext(surface_name)
        temp_name = basename + '_temp.vtk'

        print('Extracting surface from ', sdf_name)

        dfield = sitk.Cast(sitk.ReadImage(sdf_name), sitk.sitkFloat32)
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
        quiet = True
        if quiet:
            output_pipe = open(os.devnull, 'w')       # Ignore text output from MRF.exe.
        else:
            output_pipe = None        
        
        subprocess.call([self.mrf_exe, '-i', sdf_name, '-o', temp_name, '-l', '-E', str(edge_length)],stdout=output_pipe)

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
        t_x, t_y, t_z = -origin[0], -origin[1], origin[2] + spacing[2] * (size[2]-1)
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
        writer.SetFileName(surface_name)
        writer.Write()

        os.remove(temp_name)

    # @staticmethod
    # def write_landmarks_as_vtk_points(landmarks, file_name):
    #   Utils3D.write_landmarks_as_vtk_points_external(landmarks, file_name)

    # @staticmethod
    # def write_landmarks_as_text(landmarks, file_name):
    #    Utils3D.write_landmarks_as_text_external(landmarks, file_name)

    # @staticmethod
    # def visualise_mesh_and_landmarks(mesh_name, landmarks=None):
    #   Render3D.visualise_mesh_and_landmarks(mesh_name, landmarks)
