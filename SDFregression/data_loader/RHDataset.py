import imageio as imageio
from torch.utils.data import Dataset
import os
import numpy as np
#from skimage import transform
import nibabel as nib
import SimpleITK as sitk

class ElasticallyDeformImage(object):
    """
    Deform a sitk image using BSpline transformations.
    Implements elastic deformations that are used for augmentation, using the pre-made SimpleITK library.
    The number of control points is (size of the grid - 2) due to borders, while std_dev is the standard
    deviation of the displacement vectors in pixels.
    https://raw.githubusercontent.com/frapa/tbcnn/master/deformations.py
    """

    def __init__(self, num_control_points=5, std_dev=1, default_lab_val=0, default_ct_val=-1):
        self.num_control_points = num_control_points
        self.std_dev = std_dev
        self.default_lab_val = default_lab_val
        self.default_ct_val = default_ct_val

    def __call__(self, np_image, np_label):

        sitk_image = sitk.GetImageFromArray(np_image)
        sitk_lab = sitk.GetImageFromArray(np_label)

        # Allocate memory for transform parameters
        transform_mesh_size = [self.num_control_points] * sitk_image.GetDimension()
        transform = sitk.BSplineTransformInitializer(
            sitk_image,
            transform_mesh_size
        )

        # Read the parameters as a numpy array, then add random
        # displacement and set the parameters back into the transform
        params = np.asarray(transform.GetParameters(), dtype=np.float64)
        params = params + np.random.randn(params.shape[0]) * self.std_dev
        transform.SetParameters(tuple(params))

        # Create resampler object
        # The interpolator can be set to sitk.sitkBSpline for cubic interpolation,
        # see https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5 for more options
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(self.default_ct_val)
        resampler.SetTransform(transform)

        # Execute augmentation
        sitk_augmented_image = resampler.Execute(sitk_image)
        np_augmented_image = sitk.GetArrayFromImage(sitk_augmented_image)

        resampler.SetDefaultPixelValue(self.default_lab_val)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        sitk_augmented_lab = resampler.Execute(sitk_lab)
        np_augmented_lab = sitk.GetArrayFromImage(sitk_augmented_lab)

        return np_augmented_image, np_augmented_lab

class RHDataset(Dataset):
    """
    MMWHS dataset loader
    Class inspired from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, data_list, image_dir, label_dir, image_size=256, n_classes = 2, tfrm=None):
        """
        Args:
            data_list (string): Path to the csv/txt file with file ids.
            image_dir (string): Root directory for image data.
            label_dir (string): Root directory for label data.
            tfrm (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_ids = np.loadtxt(data_list, dtype=str, comments="#", delimiter=",", unpack=False)
        print('Read ', len(self.file_ids), ' file ids')

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = tfrm
        self.image_size = image_size
        self.n_channels  = 1
        self.n_classes = n_classes


    def _check_if_valid_file(self, file_name):
        if not os.path.isfile(file_name):
            print(file_name, " is not a file!")
            return False
        elif os.stat(file_name).st_size < 10:
            print(file_name, " is not valid (length less than 10 bytes)")
            return False
        return True

    def _check_image_files(self):
        # TODO: tilpas til RHdata eller slet!
        rendering_type = self.image_channels
        print('Checking if all files are there')
        new_id_table = []
        for file_name in self.file_ids:
            
            if rendering_type == 'geometry':
                image_file = os.path.join(self.root_dir, 'images', file_name + '_geometry.png')
                if self._check_if_valid_file(image_file):
                    new_id_table.append(file_name)
            elif rendering_type == 'depth':
                image_file = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
                if self._check_if_valid_file(image_file):
                    new_id_table.append(file_name)
            elif rendering_type == 'RGB':
                image_file = os.path.join(self.root_dir, 'images', file_name + '.png')
                if self._check_if_valid_file(image_file):
                    new_id_table.append(file_name)
            elif rendering_type == 'RGB+depth':
                image_file = os.path.join(self.root_dir, 'images', file_name + '.png')
                image_file2 = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
                if self._check_if_valid_file(image_file) and self._check_if_valid_file(image_file2):
                    new_id_table.append(file_name)
            elif rendering_type == 'geometry+depth':
                image_file = os.path.join(self.root_dir, 'images', file_name + '_geometry.png')
                image_file2 = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
                if self._check_if_valid_file(image_file) and self._check_if_valid_file(image_file2):
                    new_id_table.append(file_name)

        print('Checking done')
        self.file_ids = new_id_table
        print('Final ', len(self.id_table), ' file ids including augmentations')


    def _safe_read_and_scale_image(self, image_file, label_file, img_size):
        """
        """

        img_in = None

        if self._check_if_valid_file(image_file):
            try:
                img_itk = sitk.Cast(sitk.ReadImage(image_file), sitk.sitkFloat32)
                img_in = sitk.GetArrayFromImage(img_itk)
                
                label_itk = sitk.Cast(sitk.ReadImage(label_file), sitk.sitkFloat32)
                label_in = sitk.GetArrayFromImage(label_itk)
                
                org_size = img_itk.GetSize()[0]
                
                if not org_size == img_size:
                    img_in = self.resample_sitk(img_itk, img_size, interpolator = sitk.sitkLinear)    
                    label_in = self.resample_sitk(label_itk, img_size, sitk.sitkNearestNeighbor)

                img_in = np.clip(img_in,-1000,1000)/1000                    
#                label_X = np.zeros(label_in.shape).astype(np.float32)
#                label_X[label_in == 3] = 1 #LA
#                label_X[label_in == 9] = 1 #LAA
                        
#                # TODO: Only for debugging! pls remove
#                debug_dir = 'C:/Users/kajul/Documents/LAA/Segmentation/PyTorch3DUnet/3D-Unet/debug/'
#                if not os.path.exists(debug_dir):
#                    os.mkdir(debug_dir)  
#                
#                img_sitk = sitk.GetImageFromArray(img_in)
#                sitk.WriteImage(img_sitk,debug_dir+os.path.split(image_file)[1])
#                lab_sitk = sitk.GetImageFromArray(label_X)
#                sitk.WriteImage(lab_sitk,debug_dir+os.path.split(label_file)[1])
#                
#                print("Writing debugging image and label")
                
            except IOError as e:
                print("File ", image_file, " raises exception")
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except ValueError:
                print("File ", image_file, " raises exception")
                print("ValueError")
        return img_in, label_in, org_size
    
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
        
    
    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        # print('Returning item ', idx)
        file_name = self.file_ids[idx]
        
#        print("filename: ", file_name)
        
        image_file = os.path.join(self.image_dir,file_name+'.nii')
        label_file = os.path.join(self.label_dir,file_name+'.nii.gz')
               
        # Resize to input-size
        img_in, label_in, orig_size = self._safe_read_and_scale_image(image_file, label_file, self.image_size)

        if self.transform:
            img_in, label_in = self.transform(img_in, label_in)
        
        image = img_in.reshape(self.image_size, self.image_size, self.image_size, self.n_channels)
        
        if self.n_classes == 2: #(Background + foreground)
            label = np.stack((1-label_in, label_in), axis=3)
        else: 
            label = np.stack((label_in==0, label_in==1, label_in==2, label_in==3, label_in==4), axis=3).astype(np.float32)
        
        sample = {'image': image, 'label': label}


        return sample
