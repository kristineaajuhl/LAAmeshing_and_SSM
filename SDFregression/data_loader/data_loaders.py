import os

from base import BaseDataLoader
from data_loader.MMWHSDataset import MMWHSDataset
from data_loader.MMWHSDataset import ElasticallyDeformImage
from torchvision import transforms
from data_loader.RHDataset import RHDataset
from data_loader.RHDataset_dfield import RHDataset_dfield
from data_loader.RHDataset import ElasticallyDeformImage

class MMWHSDataLoader(BaseDataLoader):
    """
    Loading MMWH dataset
    """
    def __init__(self, data_list, image_dir, label_dir, image_size=64,n_classes=2, batch_size=1,
                 shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        #tfrm = ElasticallyDeformImage(5, std_dev = 2)
        tfrm = None
        
        self.dataset = MMWHSDataset(data_list, image_dir, label_dir, image_size, n_classes, tfrm = tfrm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class RHDataLoader_ROI(BaseDataLoader):
    """
    Loading ct-scans from RH
    """
    def __init__(self, data_list, image_dir, label_dir, image_size=64,n_classes=2, batch_size=1,
                 shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        #tfrm = ElasticallyDeformImage(5, std_dev = 2)
        tfrm = None
        
        self.dataset = RHDataset(data_list, image_dir, label_dir, image_size, n_classes, tfrm = tfrm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class RHDataLoader_SDF(BaseDataLoader):
    """
    Loading ct-scans from RH
    """
    def __init__(self, data_list, image_dir, label_dir, image_size=64,n_classes=2, batch_size=1,
                 shuffle=True, validation_split=0.0, num_workers=1, training=True):
        
        #tfrm = ElasticallyDeformImage(5, std_dev = 2)
        tfrm = None
        
        self.dataset = RHDataset_dfield(data_list, image_dir, label_dir, image_size, n_classes, tfrm = tfrm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        