import ast
import argparse
import pandas         as     pd


# Read arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input_filename',  type=str, default='data/data_script2.xlsx')
parser.add_argument('--output_filename', type=str, default='data/data_script3.xlsx')
args   = parser.parse_args()

input_filename  = args.input_filename
output_filename = args.output_filename
input_data      = pd.read_excel(input_filename)
len_data        = len(input_data)

# Including DICOM scans based on the following conditions:
# Empty attributes
# scan_size >= 20
# scan_spacing[0] < 1
# scan_spacing[1] < 1
# slice thickness: scan_spacing[2] >= 1 and scan_spacing[2] <= 5  

def apply_inclusion_criteria(scan_size, scan_spacing):
    scan_size    = ast.literal_eval(scan_size)
    scan_spacing = ast.literal_eval(scan_spacing)
    
    if (scan_size[0] < 20 or scan_size[1] < 20 or scan_size[2] < 20):
        return 'excluded'
    
    elif (scan_spacing[0] < 1 and scan_spacing[1] < 1 and scan_spacing[2] >= 1 and scan_spacing[2] <= 5):
        return 'train'
    else:
        return 'test'
    

# save file without empty attribute values    
new_data = input_data.dropna(subset=['scan_origin'])
new_data.to_excel((output_filename.replace('.xlsx', 'A_all_values.xlsx')), index=False)

# save file with the flag of inclusion
new_data.insert(len(new_data.columns), 'scan_inclusion', new_data.apply(lambda x: apply_inclusion_criteria(x.scan_size, x.scan_spacing), axis=1))
new_data.to_excel((output_filename.replace('.xlsx', 'B_inclusion_values.xlsx')), index=False)

# Save file only considering the included scans
new_data = new_data[new_data['scan_inclusion'] != 'excluded']
new_data.to_excel(output_filename, index=False)