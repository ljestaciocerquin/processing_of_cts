import argparse
import pandas         as     pd
from   tqdm           import tqdm
from   cts_operations import ReadDICOM
from   cts_operations import ReadScanAttributes


# Read arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input_filename',  type=str, default='data/data_script1.xlsx')
parser.add_argument('--output_filename', type=str, default='data/data_script2.xlsx')
args   = parser.parse_args()

input_filename       = args.input_filename
output_filename      = args.output_filename
input_data           = pd.read_excel(input_filename)
list_of_scan_folders = input_data['ct_folder']
list_of_folders      = []
list_of_attributes   = []
len_data             = len(list_of_scan_folders)

# Walk through the directory structure to get the scan attributes
with tqdm(total=len_data) as pbar:
    for folder in list_of_scan_folders:
        image        = ReadDICOM()(folder)
        image_attrib = ReadScanAttributes()(image)
        list_of_folders.append(folder)
        list_of_attributes.append(image_attrib)
        print(folder)
        pbar.update(1)
list_of_attributes = pd.DataFrame.from_dict(list_of_attributes)

data_folder = {
    "scan_folder"   : list_of_folders,
}

# Create a Pandas DataFrame
df_folders   = pd.DataFrame(data_folder)
data_script2 = pd.concat([df_folders,  list_of_attributes], axis=1, ignore_index=False, sort=False) 
data_script2.to_excel(output_filename, index=False)