import numpy     as np
import SimpleITK as sitk
from cts_processors import ReadProcessor
from cts_processors import WriteProcessor
from cts_processors import AttributesProcessor
from cts_processors import TransformProcessor
from numpy import floor, ceil

class ReadVolume(ReadProcessor):
    
    def __call__(self, filename):
        try:
            image = sitk.ReadImage(fileName=filename)
            return image
        except:
            return None
    


class ReadDICOM(ReadProcessor):
    
    def __call__(self, filename):
        try:
            reader    = sitk.ImageSeriesReader()
            dcm_files = reader.GetGDCMSeriesFileNames(filename)
            reader.SetFileNames(dcm_files)
            image = reader.Execute()
            return image
        except:
            return None



class ReadScanAttributes(AttributesProcessor):
    
    def __call__(self, image):
        
        try:
            scan_origin    = image.GetOrigin()
            scan_size      = image.GetSize()
            scan_spacing   = image.GetSpacing()
            scan_direction = image.GetDirection()
            
            attributes_dict = {
                'scan_origin'   : scan_origin,
                'scan_size'     : scan_size,
                'scan_spacing'  : scan_spacing,
                'scan_direction': scan_direction,
            } 
            
            return attributes_dict
        
        except:
            attributes_dict = {
                'scan_origin'   : '',
                'scan_size'     : '',
                'scan_spacing'  : '',
                'scan_direction': '',
            } 
            
            return attributes_dict



class ResampleScan(TransformProcessor):
    
    def __init__(self, spacing=1., interpolator=sitk.sitkLinear):
        
        self.spacing      = spacing
        self.interpolator = interpolator
        self.operation    = sitk.ResampleImageFilter()
        super(ResampleScan, self).__init__()
    
    
    def __call__(self, image):
        
        spacing = self.spacing
        if not isinstance(spacing, list):
            spacing = [spacing, ] * 3
        self.operation.SetReferenceImage(image)
        self.operation.SetOutputSpacing(spacing)
        self.operation.SetInterpolator(self.interpolator)
        
        s0 = int(round((image.GetSize()[0] * image.GetSpacing()[0]) / spacing[0], 0))
        s1 = int(round((image.GetSize()[1] * image.GetSpacing()[1]) / spacing[1], 0))
        s2 = int(round((image.GetSize()[2] * image.GetSpacing()[2]) / spacing[2], 0))
        self.operation.SetSize([s0, s1, s2])
        
        return self.operation.Execute(image)

    
    
class WriteScan(WriteProcessor):
    
    def __call__(self, image, filename):
        try:
            sitk.WriteImage(image=image, fileName=filename)
        except:
            None
            
    
        
class ResampleScan(TransformProcessor):
    
    def __init__(self, spacing=1., interpolator=sitk.sitkLinear):
        
        self.spacing      = spacing
        self.interpolator = interpolator
        self.operation    = sitk.ResampleImageFilter()
        super(ResampleScan, self).__init__()
    
    
    def __call__(self, image):
        
        spacing = self.spacing
        if not isinstance(spacing, list):
            spacing = [spacing, ] * 3
        self.operation.SetReferenceImage(image)
        self.operation.SetOutputSpacing(spacing)
        self.operation.SetInterpolator(self.interpolator)
        
        s0 = int(round((image.GetSize()[0] * image.GetSpacing()[0]) / spacing[0], 0))
        s1 = int(round((image.GetSize()[1] * image.GetSpacing()[1]) / spacing[1], 0))
        s2 = int(round((image.GetSize()[2] * image.GetSpacing()[2]) / spacing[2], 0))
        self.operation.SetSize([s0, s1, s2])
        
        return self.operation.Execute(image)
    

class PadAndCropTo(TransformProcessor):
    
    def __init__(self, target_shape, cval=0.):
        self.target_shape = target_shape
        self.cval         = cval
        super(PadAndCropTo, self).__init__()

    def __call__(self, image):
        # padding
        shape        = image.GetSize()
        target_shape = [s if t is None else t for s, t in zip(shape, self.target_shape)]
        pad          = [max(s - t, 0) for t, s in zip(shape, target_shape)]
        lo_bound     = [int(floor(p / 2)) if i != 2 else 0 for i, p in enumerate(pad)]  # Set depth (third dimension) padding to 0
        up_bound     = [int(ceil(p / 2)) if i != 2 else 0 for i, p in enumerate(pad)]  # Set depth (third dimension) padding to 0
        image        = sitk.ConstantPad(image, lo_bound, up_bound, self.cval)

        # cropping
        shape        = image.GetSize()
        target_shape = [s if t is None else t for s, t in zip(shape, self.target_shape)]
        crop         = [max(t - s, 0) for t, s in zip(shape, target_shape)]
        lo_bound     = [int(floor(c / 2)) if i != 2 else 0 for i, c in enumerate(crop)]  # Set depth (third dimension) cropping to 0
        up_bound     = [int(ceil(c / 2)) if i != 2 else 0 for i, c in enumerate(crop)]  # Set depth (third dimension) cropping to 0
        image        = sitk.Crop(image, lo_bound, up_bound)

        return image



    
class RandomNoise(TransformProcessor):

    def __init__(self, gaussian=None, poisson=None, gamma=None):
        self.gaussian = gaussian
        self.poisson  = poisson
        self.gamma    = gamma
        super(RandomNoise, self).__init__()

    def __call__(self, image, *args):

        if len(args) == 0:
            args = [self.get_random_params(image)]

        transform = args[0]

        if len(args) > 1:
            for t in args[1:]:
                transform = transform + t

        transform = sitk.GetImageFromArray(transform)
        transform.SetOrigin(image.GetOrigin())
        transform.SetSpacing(image.GetSpacing())
        transform.SetDirection(image.GetDirection())
        
        # adding noise to the original image
        image = sitk.Add(image, transform)

        return image

    def get_random_params(self, image):

        np.random.seed(self.random_seed)

        s = image.GetSize()
        transform = np.zeros(s)

        if self.gaussian is not None:
            transform += np.random.normal(self.gaussian[0], self.gaussian[1], size=s)

        if self.poisson is not None:
            transform += np.random.poisson(self.poisson[0], size=s)

        if self.gamma is not None:
            transform += np.random.gamma(self.gamma[0], self.gamma[1], size=s)

        return transform
    
    
    
class ToNumpyArray(TransformProcessor):

    def __init__(self, add_batch_dim=False, add_singleton_dim=False, normalize=True):
        self.add_batch_dim     = add_batch_dim
        self.add_singleton_dim = add_singleton_dim
        self.normalize         = normalize
        super(ToNumpyArray, self).__init__()

    def __call__(self, image):
        image = sitk.GetArrayFromImage(image)
        # Normalize the values if requested
        if self.normalize:
            image = self.normalize_values(image)
        if self.add_batch_dim:
            image = image[None]
        if self.add_singleton_dim:
            image = image[..., None]
        return image
    
    def normalize_values(self, image_array):
        # min-max normalization to scale values between 0 and 1
        min_value = np.min(image_array)
        max_value = np.max(image_array)
        normalized_array = (image_array - min_value) / (max_value - min_value)
        return normalized_array