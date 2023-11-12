import SimpleITK as sitk
import multiprocessing
import logging
from SimpleITK import Image

class ScanProcessor(object):
    
    def __init__(self, *steps):
        self.steps = steps
        super(ScanProcessor, self).__init__()
        
    
    def __call__(self, inputs):
        for s in self.steps:
            inputs = s(inputs)
        return inputs
   
    

class ReadProcessor(object):
    
    def __call__(self, filename):
        return Image()



class AttributesProcessor(object):
    
    def __call__(self, image):
        return dict()



class ResamplingProcessor(object):
    
    def __call__(self, image):
        return Image()
    


class WriteProcessor(object):
    
    def __call__(self, image, filename):
        return 
    
#logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
#logging.info(str(args))
