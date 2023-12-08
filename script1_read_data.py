import os
import argparse
import pandas   as pd

# Read arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input_folder',    type=str, default='/data/groups/beets-tan/_public/tcia/CancerImagingArchive_20200421/CT/TCGA-KIRC')
parser.add_argument('--output_filename', type=str, default='data/data_script1.xlsx')
args   = parser.parse_args()

input_folder    = args.pancancer_folder
output_filename = args.output_filename
dcm_folders     = []


def contains_dicom_files(directory):
    for file in os.listdir(directory):
        if file.endswith(".dcm"):
            return True
    return False

# Walk through the directory structure starting from the root folder
for root, dirs, files in os.walk(input_folder):
    for dir_name in dirs:
        path_to_folder = os.path.join(root, dir_name)
        if contains_dicom_files(path_to_folder):
            print("Path found:", path_to_folder)
            dcm_folders.append(path_to_folder)

data = {
    "ct_folder": dcm_folders,
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)
df.to_excel(output_filename, index=False)