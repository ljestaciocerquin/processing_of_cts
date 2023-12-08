import os
import wandb
import torch
import argparse
import pandas             as pd
import torch.nn           as nn
from   pathlib        import Path
from   PIL                          import Image, ImageOps
import torchvision.transforms       as     T
import numpy as np

def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Directory created in: ", path)
    else:
        print("Directory already created: ", path)


def cuda_seeds(seed):
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark    = False


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        m.weight.data = nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find("Linear") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
            

def read_train_data(path_input):
    data = pd.read_excel(path_input)
    #data = data[:60]
    train_data = data[data['scan_inclusion'] == 'train']
    return train_data


def read_train_data_from_file(filename):
    data = pd.read_csv(filename)
    print('total len: ', len(data))
    train_data = data.loc[data['fold'] == 'train']
    train_data = train_data[['PRIOR_PATH_NRRD', 'SUBSQ_PATH_NRRD']]
    print('only training len: ', len(train_data))
    valid_data = data.loc[data['fold'] == 'valid']
    valid_data = valid_data[['PRIOR_PATH_NRRD', 'SUBSQ_PATH_NRRD']]
    print('only validation len: ', len(valid_data))
    return train_data, valid_data


def save_images_weights_and_biases(table_name, path_to_save, fixed, moving, w0_img, w1_img):
    table = wandb.Table(columns=['Fixed Image', 'Moving Image', 'Affine Reg. Image', 'Deformation Reg. Image'], allow_mixed_types = True)
    
    saving_examples_folder = path_to_save
    
    print('Fixed: ', fixed.shape)
    print('w_0: ', w0_img.shape)
    #PIL VERSION
    transform  = T.ToPILImage() 
    fixed_img  = transform(fixed[:,:,:,50].squeeze()).convert("L") 
    moving_img = transform(moving[:,:,:,50].squeeze()).convert("L")
    affine_img = transform(w0_img[0,:,:,50].squeeze()).convert("L") # The 0 is in order to visualize the deformation field transform(w0_img[0,:,:,50].squeeze()).convert("L")
    deformation_img   = transform(w1_img[:,:,:,50].squeeze()).convert("L")
    #deformation_field = transform(t1_img[:,:,:,50].squeeze()).convert("L")
    
    fixed_img.show()                              
    fixed_img.save(saving_examples_folder + "fixed_image.png")    
    moving_img.show() 
    moving_img.save(saving_examples_folder + "moving_image.png")    
    affine_img.show() 
    affine_img.save(saving_examples_folder + "affine_image.png")    
    deformation_img.show()
    deformation_img.save(saving_examples_folder + "deformation_image.png")  
    #deformation_field.show()
    #deformation_field.save(saving_examples_folder + "deformation_field_image.png")  
    
    table.add_data(
        wandb.Image(Image.open(saving_examples_folder + "fixed_image.png")),
        wandb.Image(Image.open(saving_examples_folder + "moving_image.png")),
        wandb.Image(Image.open(saving_examples_folder + "affine_image.png")),
        wandb.Image(Image.open(saving_examples_folder + "deformation_image.png"))
        #wandb.Image(Image.open(saving_examples_folder + "deformation_field_image.png"))
    )
    
    wandb.log({table_name: table})
    

def save_images_weights_and_biases_dnet(table_name, path_to_save, clean_scan, noisy_scan, predicted_scan, epoch):
    table = wandb.Table(columns=['Clean Scan', 'Noisy Scan', 'Predicted scan'], allow_mixed_types = True)
    saving_examples_folder = path_to_save
    
    #print(clean_scan.shape)
    #print(noisy_scan.shape)
    #print(predicted_scan.shape)

    #PIL VERSION
    transform     = T.ToPILImage() 
    clean_img     = transform(clean_scan[:,19,:,:].squeeze()).convert("L") 
    noisy_img     = transform(noisy_scan[:,19,:,:].squeeze()).convert("L")
    predicted_img = transform(predicted_scan[:,19,:,:].squeeze()).convert("L") # The 0 is in order to visualize the deformation field transform(w0_img[0,:,:,50].squeeze()).convert("L")
    
    clean_img_path = saving_examples_folder + 'clean_img_' + str(epoch) + '.png'
    noisy_img_path = saving_examples_folder + 'noisy_img_' + str(epoch) + '.png'
    predicted_img_path = saving_examples_folder + 'predicted_img_' + str(epoch) + '.png'
    
    clean_img.show()                              
    clean_img.save(clean_img_path)    
    noisy_img.show() 
    noisy_img.save(noisy_img_path)    
    predicted_img.show() 
    predicted_img.save(predicted_img_path)    
    
    table.add_data(
        wandb.Image(Image.open(clean_img_path)),
        wandb.Image(Image.open(noisy_img_path)),
        wandb.Image(Image.open(predicted_img_path))
    )
    wandb.log({table_name: table})