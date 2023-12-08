from cts_processors import ScanProcessor
from cts_processors import ReadProcessor
from cts_operations import ReadDICOM
from cts_operations import ReadScanAttributes
from cts_operations import ResampleScan
from cts_operations import WriteScan
import multiprocessing

def main():
    dcm_files = ['/projects/disentanglement_methods/data/Liver-Kidney/4.000000-KID DELAY-21162',
                 '/projects/disentanglement_methods/data/Liver-Kidney/6.000000-KID DELAY-02818']
    
    # First way to build the set of operations
    '''scan_process = ScanProcessor(
        ReadDICOM(),
        ReadScanAttributes(),
        #WriteScan()
    )'''
    #results = scan_process(dcm_files[0])
    #print(results)
    
    # Second way to build the set of operations
    image        = ReadDICOM()(dcm_files[0])
    image_attrib = ReadScanAttributes()(image)
    resam_image  = ResampleScan()(image)
    new_attrib   = ReadScanAttributes()(resam_image)
    print(image_attrib)
    print(new_attrib)
    

if __name__ == '__main__':
    main()