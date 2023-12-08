from   handler_VGG import Handler
import argparse

def main(args):
    
    handler = Handler(args=args)
    handler.run()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Registration of Medical Images for Prognostic Monitoring')
    
    # Handler 
    parser.add_argument('--task',        type=str,  default='train',    help='task to be performed train-pam, train-adv-pam, test or visualization')
    
    # Data
    parser.add_argument('--output_dir',  type=str, default='/projects/disentanglement_methods/processing_of_cts/output-vgg/',     help='folder to save all outputs from train | test | survival |visualization')
    parser.add_argument('--data_path' ,  type=str, default='/projects/disentanglement_methods/processing_of_cts/data/data_script5.xlsx', help='path that contains the data. It could be a folder(tcia) or a csv file (nki)') # /processing/l.estacio/data_tcia/train/
    parser.add_argument('--exp_name_wb', type=str, default='DenoiseCTNet',                                   help='experiment name weights and biases')
    parser.add_argument('--entity_wb',   type=str, default='ljestaciocerquin',                                  help='Entity for weights and biases')
    parser.add_argument('--show_img_wb', type=bool,  default=True,                                              help='it shows images when the dataset is open source: True | False')
    
    # Networks
    parser.add_argument('--input_dim',   type=int, default=[None, None, None],                 help='image dimension - nrrd')
    parser.add_argument('--filters',     type=int, default=[16, 32, 64, 128],              help='filters to create the Affine and Deformation networks') #[16, 32, 64, 128]
    parser.add_argument('--input_ch',    type=int, default=1,                               help='number of input channels for the affine and deformation networks')
    parser.add_argument('--output_ch',   type=int, default=1,                               help='number of output channels of the deformation field')
    parser.add_argument('--group_num',   type=int, default=8,                               help='group normalization size')
    parser.add_argument('--filters_disc',type=int, default=[32, 64, 128],               help='filters to create the Discriminator network')
    parser.add_argument('--input_ch_disc',type=int, default=1,                               help='number of input channels for the discriminator network')
    parser.add_argument('--filters_vgg',type=int, default=[64, 128],               help='filters to create the Discriminator network')
    parser.add_argument('--out_fts_vgg',type=int, default=128,                               help='number of input channels for the discriminator network')
    
    # Train
    parser.add_argument('--seed',           type=int,   default=42,             help='random seed')
    parser.add_argument('--cuda',           type=bool,  default=True,           help='enable cuda')
    parser.add_argument('--num_gpus',       type=int,   default=1,              help='number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument('--num_workers',    type=int,   default=8,              help='dataloader num_workers')
    parser.add_argument('--start_ep',       type=int,   default=1,             help='start training iteration')
    parser.add_argument('--n_epochs',       type=int,   default=400,            help='maximum training iteration')
    parser.add_argument('--batch_size',     type=int,   default=1,              help='batch size')
    parser.add_argument('--train_split',    type=float, default=0.8,            help='percentage to split the dataset for training')
    parser.add_argument('--alpha_value',    type=float, default=0.0,           help='beta parameter for the affine penalty loss')
    parser.add_argument('--beta_value',     type=float, default=0.1,           help='beta parameter for the deformation penalty loss')
    parser.add_argument('--gamma_value',    type=float, default=0.1,            help='gamma parameter for the discriminator (feature matching loss: MSE)')
    parser.add_argument('--lr',             type=float, default=1e-3,           help='learning rate 3e-4 ')
    parser.add_argument('--beta1',          type=float, default=0.5,            help='adam optimizer beta1')
    parser.add_argument('--beta2',          type=float, default=0.999,          help='adam optimizer beta2')
    
    # Re-train
    parser.add_argument('--re_train',       type=bool,  default=False,           help='re-train a model')
    parser.add_argument('--chk_gen_to_load',type=str,   default='',  help='generator checkpoint to be load when we re-train the model')
    parser.add_argument('--chk_dis_to_load',type=str,   default='',  help='discriminator checkpoint to be load when we re-train the model')
    parser.add_argument('--chk_vgg_to_load',type=str,   default='',  help='vgg checkpoint to be load when we re-train the model')
    
    # Test: This variables are None as default in train
    parser.add_argument('--need_ls_z',            type=bool, default=True, help='This flag helps to call to another forward which gives the latent space vector')
    parser.add_argument('--save_outputs_as_nrrd', type=bool, default=True, help='This flag helps to save all outputs (w_0 | w_1 | t_0 | t_1) inside a folder as .nrrd')
    
    # Visualization
    #feature_pos, scale_factor
    
    args = parser.parse_args()
    
    main(args)
    