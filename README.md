# LAA segmentation, mesh modelling and statistical shape modelling
This repository contains the implementations from the paper: "XXX". 

# Image segmentation and mesh model creation
Download models:  
Download the models from the URL below and place them in "/saved/trained/"  
XXX Make link to pretrained models  

Train new models:  
COMING UP

Predict ROI  
Takes the input images from the "img" folder in the same folder as the filelist and saves a lowresolution label (lowres_label) and the cropped image (img) in the ROI subfolder.  
```
python predictROI.py --c <config_file.json> --n <path_to_file.nii> (--d <device>)  
python predictROI.py --c <config_file.json> --n <path_to_filelist.txt> (--d <device>)  
```

Predict SDF  
Takes the cropped images from the ROI/img folder and saves the predicted label (.nii), distance field (.nii) and surface model (.vtk) in the Predictions subfolder  
```
python predictSDF.py --c <config_file.json> --n <path_to_file.nii> (--d <device>)  
python predictSDF.py --c <config_file.json> --n <path_to_filelist.txt> (--d <device>)  
```

Train new models:  
COMING UP

# LAA decoupling
Create distance fields  
Creates a SDF from all surfaces in the filelist.txt in the folder surfacepath  
```
python create_SDF.py --n <path_to_filelist.txt> --s <path_to_surfaces> --p <"Full"/"LAA">  
```

Register to common template  
Registers all LAs to a common template (point correspondence). Use precomputed template (0) or make new one from 3 iterations on your data (1)  
```
python register_to_template.py --n <path_to_filelist.txt> --s <path_to_surfaces> --p <"Full"/"LAA"> --t <0/1>  
```

Decouple LAA from remaining LA  
Decouples the LAA from the remaining LA and assigns five anatomical landmarks with or without visualizing the process  
```
python cut_all_examples.py --n <path_to_filelist.txt> --s <path_to_original_surfaces> --c <path_to_surfaces_in_correspondence> --v <True/False>  
```

# LAA shape model
COMING UP 

# Dependencies
MRFtools  
elastix

# Directory setup:
  * filelist.txt  
  * img  
  * lab  
  * surf_models  
  * ROI  
    * lowres_label  
    * img  
$\qquad$  -- lab  
$\qquad$  --dfield  
-- Predictions  
$\qquad$  -- lab  
$\qquad$  -- dfield  
$\qquad$ -- surf_models  
$\qquad$  -- LAA decoupling  
>>>    -- distance_filds  
>>>    -- LAA_LM  
>>>    -- LAA_only  
>>>    -- output  
