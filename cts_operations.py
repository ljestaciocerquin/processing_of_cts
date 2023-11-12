import SimpleITK as sitk
from cts_processors import ReadProcessor
from cts_processors import WriteProcessor
from cts_processors import AttributesProcessor
from cts_processors import ResamplingProcessor


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



class ResampleScan(ResamplingProcessor):
    
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
