import os
import argparse
import pandas         as     pd
from   tqdm           import tqdm
from   cts_operations import ReadDICOM, ReadVolume
from   cts_operations import WriteScan
from   cts_operations import ReadScanAttributes

# Read arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input_filename',  type=str, default='data/data_script5.xlsx')
parser.add_argument('--output_filename', type=str, default='data/data_script6.xlsx')
args   = parser.parse_args()

input_filename  = args.input_filename
output_filename = args.output_filename
input_data      = pd.read_excel(input_filename)

# Replace the specified part of the path
selected_rows = input_data[input_data['scan_size'].apply(lambda x: int(x.split(',')[2][:-1]) < 100)]
selected_rows.to_excel(output_filename, index=False)