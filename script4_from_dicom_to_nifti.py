import os
import argparse
import pandas         as     pd
from   tqdm           import tqdm
from   cts_operations import ReadDICOM, ReadVolume
from   cts_operations import WriteScan
from   cts_operations import ReadScanAttributes

# Read arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input_filename',  type=str, default='data/data_script3.xlsx')
parser.add_argument('--output_filename', type=str, default='data/data_script4.xlsx')
parser.add_argument('--folder_to_save',  type=str, default='/data/groups/beets-tan/l.estacio/reg_data/')
args   = parser.parse_args()

input_filename  = args.input_filename
output_filename = args.output_filename
folder_to_save  = args.folder_to_save
input_data      = pd.read_excel(input_filename)
len_data        = len(input_data)

# Adding scan_id column
input_data.insert(0, 'scan_id', range(1, len(input_data) + 1))


# Function to make folders
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Directory created in: ", path)
    else:
        print("Directory already created: ", path)
      
        
# Create imagesTs and labelsTs folders
train_folder = folder_to_save + 'train/'
test_folder  = folder_to_save + 'test/'
create_directory(train_folder)
create_directory(test_folder)


# From DICOM to Nifti
path_nifti = []
with tqdm(total=len_data) as pbar:
    for idx, row in input_data.iterrows():
        scan_id               = row['scan_id']
        scan_folder           = row['scan_folder']
        scan_inclusion_folder = row['scan_inclusion']
        print(scan_folder)
        # Reading DICOM
        image       = ReadDICOM()(scan_folder)
        
        # Saving Nifti
        nifti_scan_name = folder_to_save + scan_inclusion_folder + '/' + str(scan_id) + '-' +  scan_folder.split('/')[8] + '-' + scan_folder.split('/')[9] + '.nii.gz'
        WriteScan()(image, nifti_scan_name)
        path_nifti.append(nifti_scan_name)
        pbar.update(1)

input_data['scan_nifti_path'] = path_nifti
input_data.to_excel(output_filename, index=False)

'''data_nifti = pd.read_excel(output_filename)
with tqdm(total=len_data) as pbar:
    for idx, row in data_nifti.iterrows():
        scan_folder           = row['scan_nifti_path']
        scan_id               = row['scan_id']
        print(scan_folder)
        # Reading DICOM
        image       = ReadVolume()(scan_folder)
        image_attrib = ReadScanAttributes()(image)
        
        print(scan_id, ': ', image_attrib)'''