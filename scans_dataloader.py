import torch
import random
import numpy        as     np
from torch.utils    import data
from cts_processors import ScanProcessor
from cts_operations import ReadVolume
from cts_operations import PadAndCropTo
from cts_operations import ToNumpyArray


class ScanDataLoader(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 input_dim   : int    = [None, None, None],
                 transform   : object = None
                 ):
        self.dataset     = path_dataset
        self.input_shape = tuple(input_dim + [1]) # Giving the right shape as (192, 192, 160, 1)
        self.transform   = transform
        self.inp_dtype   = torch.float32
        self.loader      = self.__init_operations()

        
    def __init_operations(self):
        return ScanProcessor(
            ReadVolume(),
            #PadAndCropTo((128, 128 , None)),
            ToNumpyArray()
        )

    def __add_noise(self, image):
        
        np.random.seed(42)
        s            = image.shape
        transform    = np.zeros(s)
        distribution = random.choice(['gaussian', 'poisson'])
        
        if distribution == 'gaussian':
            std = round(random.uniform(0, 0.5), 2)
            transform += np.random.normal(0, std, size=s)

        if distribution == 'poisson':
            rate = round(random.uniform(0, 0.5), 2)
            transform += np.random.poisson(rate, size=s)
        
        image = image + transform
        image = np.clip(image, 0, 1)
        return image
        
        
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index: int):

        # Select the sample
        #print("Image Path: ", self.dataset.iloc[index])
        scan_path  = str(self.dataset.iloc[index]['scan_nifti_path']) 
        scan_image = self.loader(scan_path)
        scan_noise = self.__add_noise(scan_image)

        #fx = fx.transpose(1, 2, 0)
        scan_image = torch.from_numpy(scan_image).type(self.inp_dtype)
        scan_noise = torch.from_numpy(scan_noise).type(self.inp_dtype)
        scan_image = scan_image[None, :]
        scan_noise = scan_noise[None, :]
        #print('Scan clean size: ', scan_image.shape, ' min: ', scan_image.min(), ' max: ', scan_image.max())
        #print('Scan noise size: ', scan_noise.shape, ' min: ', scan_noise.min(), ' max: ', scan_noise.max())
        return scan_image, scan_noise