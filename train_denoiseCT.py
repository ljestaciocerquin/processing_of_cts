import os
import torch
import wandb
from   utils                        import create_directory
from   utils                        import cuda_seeds
from   utils                        import weights_init
from   utils                        import read_train_data
from   utils                        import save_images_weights_and_biases_dnet
from   scans_dataloader       import ScanDataLoader
from   early_stopping               import EarlyStopping
from   denoiseCT_net                                    import DenoiseCT_Net
from   denoiseCT_net                                    import Discriminator
from denoiseCT_net import VGG4
from   torch.utils.data                            import DataLoader
from   sklearn.model_selection                     import train_test_split
from tqdm import tqdm
from metrics import calculate_gradient_penalty

class Train_DenoiseCT(object):
    def __init__(self, args):
        
        # Handler
        self.task        = args.task
        
        # Data
        self.output_dir  = args.output_dir
        self.data_path   = args.data_path
        self.exp_name_wb = args.exp_name_wb
        self.entity_wb   = args.entity_wb
        
        # Networks
        self.input_dim   = args.input_dim
        self.filters     = args.filters
        self.input_ch    = args.input_ch
        self.output_ch   = args.output_ch
        self.group_num   = args.group_num
        self.filters_disc  = args.filters_disc
        self.input_ch_disc = args.input_ch_disc
        self.filters_vgg   = args.filters_vgg
        self.out_fts_vgg   = args.out_fts_vgg
        
        # Train
        self.seed          = args.seed
        self.cuda          = args.cuda
        self.num_gpus      = args.num_gpus
        self.num_workers   = args.num_workers
        self.start_ep      = args.start_ep
        self.n_epochs      = args.n_epochs
        self.batch_size    = args.batch_size
        self.train_split   = args.train_split
        self.alpha_value   = args.alpha_value
        self.beta_value    = args.beta_value
        self.gamma_value   = args.gamma_value
        self.lr            = args.lr
        self.beta1         = args.beta1
        self.beta2         = args.beta2
        self.re_train      = args.re_train
        self.show_img_wb   = args.show_img_wb
        self.chk_gen_to_load = args.chk_gen_to_load
        self.chk_dis_to_load = args.chk_dis_to_load
        self.chk_vgg_to_load = args.chk_vgg_to_load
        
        # Variables to save weights and biases images
        if self.show_img_wb: 
            self.clean_scan     = None
            self.noisy_scan     = None
            self.predicted_scan = None
        
        
    def save_outputs(self):
        # Directory to save checkpoints
        self.checkpoints_folder = self.output_dir + self.task + '/' + 'checkpoints/'
        self.results_dir        = self.output_dir + self.task + '/' + 'images-val/'
        create_directory(self.checkpoints_folder)
        create_directory(self.results_dir)
        
        
    def load_model_weights(self):
        # Loading the model weights
        gen_chkpt = self.checkpoints_folder + self.chk_gen_to_load
        dis_chkpt = self.checkpoints_folder + self.chk_dis_to_load
        #vgg_chkpt = self.checkpoints_folder + self.chk_vgg_to_load
        self.gen_net.load_state_dict(torch.load(gen_chkpt))
        self.dis_net.load_state_dict(torch.load(dis_chkpt))
        #self.vgg_net.load_state_dict(torch.load(vgg_chkpt))


    def model_init(self):
        # cuda seeds
        cuda_seeds(self.seed)
        
        # Device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.num_gpus > 0) else "cpu")
        
        # Network Definitions to the device
        self.gen_net = DenoiseCT_Net(self.input_ch, self.input_dim, self.output_ch, self.group_num, self.filters)
        self.dis_net = Discriminator(self.input_ch_disc, self.input_dim, self.group_num, self.filters_disc)
        #self.vgg_net = VGG4(self.input_ch_disc, self.input_dim, self.group_num, self.filters_vgg, self.out_fts_vgg)
        self.gen_net.to(self.device)
        self.dis_net.to(self.device)
        #self.vgg_net.to(self.device)
        
        if self.re_train:
            print('Loading pretrained weights')
            self.load_model_weights()
        else:
            self.gen_net.apply(weights_init)
            self.dis_net.apply(weights_init)
            #self.vgg_net.apply(weights_init)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.num_gpus > 1):
            self.gen_net = torch.nn.DataParallel(self.gen_net, list(range(self.num_gpus)))
            self.dis_net = torch.nn.DataParallel(self.dis_net, list(range(self.num_gpus)))
            #self.vgg_net = torch.nn.DataParallel(self.vgg_net, list(range(self.num_gpus)))

    
    def init_loss_functions(self):
        self.disc_loss    = torch.nn.BCELoss()  
        self.gen_loss     = torch.nn.MSELoss()  
        #self.vgg_loss     = torch.nn.MSELoss() 


    def set_optimizer(self):
        self.gen_optimizer = torch.optim.Adam(self.gen_net.parameters(), lr = self.lr, betas=(self.beta1, self.beta2)) #lr =0.0002
        self.dis_optimizer = torch.optim.Adam(self.dis_net.parameters(), lr = self.lr, betas=(self.beta1, self.beta2))
        #self.vgg_optimizer = torch.optim.Adam(self.vgg_net.parameters(), lr = self.lr, betas=(self.beta1, self.beta2))
        
                    
    def load_dataloader(self):
        # Dataset Path 
        filenames                  = read_train_data(self.data_path)
        inputs_train, inputs_valid = train_test_split(filenames, random_state=self.seed, train_size=self.train_split, shuffle=True)
        print("total: ", len(filenames), " train: ", len(inputs_train), " valid: ", len(inputs_valid))
        
        # Training and validation dataset
        train_dataset = ScanDataLoader(path_dataset = inputs_train,
                                       input_dim    = self.input_dim,
                                       transform    = None)
        valid_dataset = ScanDataLoader(path_dataset = inputs_valid,
                                       input_dim    = self.input_dim,
                                       transform    = None)
        # Training and validation dataloader
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    
    def train_one_epoch(self):
        loss_gen_train  = 0
        loss_disc_train = 0
        loss_vgg_train  = 0
        self.gen_net.train()
        self.dis_net.train()
        #self.vgg_net.train()

        for i, (x_1, x_2) in enumerate(self.train_dataloader):
            # send to device (GPU or CPU)
            clean_scan  = x_1.to(self.device)
            noisy_scan  = x_2.to(self.device)
            
            # (1) Update Discriminator
            self.dis_optimizer.zero_grad()
            real_output = self.dis_net(clean_scan)
            errD_real   = torch.mean(real_output)
            D_x         = real_output.mean().item()
            
            # Generate fake images
            fake_images = self.gen_net(noisy_scan)
            
            # Train with fake
            fake_output = self.dis_net(fake_images.detach())
            errD_fake   = torch.mean(fake_output)
            D_G_z1      = fake_output.mean().item()
            
            # Calculate W-div gradient penalty
            gradient_penalty = calculate_gradient_penalty(self.dis_net, clean_scan.data, fake_images.data, self.device)
            
            # Add the gradients from the all-real and all-fake batches
            errD = -errD_real + errD_fake + gradient_penalty * 10
            loss_disc_train += errD.item()
            
            # Update D
            errD.backward()
            self.dis_optimizer.step()
            
            
            
            # Train the generator every n_critic iterations
            
            #if (i + 1) % self.n_critic == 0:
            # (2) Update G network
            self.gen_optimizer.zero_grad()
            
            # Generate fake images
            fake_images = self.gen_net(noisy_scan)
            fake_output = self.dis_net(fake_images)
            errG_bce    = -torch.mean(fake_output)
            errG_L2     = self.gen_loss(fake_images, clean_scan)
            errG        = errG_bce + errG_L2
            loss_gen_train += errG.item()
            
            D_G_z2      = fake_output.mean().item()
            errG.backward()
            self.gen_optimizer.step()
            
            # Reinit the affine network weights
            if errD.item() < 1e-5:  # >
                self.dis_net.apply(weights_init)
                print("Reloading discriminator weights")
            
            
            # Train VGG
            '''self.vgg_optimizer.zero_grad()
            fake_images = self.gen_net(noisy_scan)
            real_fts = self.vgg_net(clean_scan)
            fake_fts = self.vgg_net(fake_images)
            errVGG   = self.vgg_loss(fake_fts, real_fts)
            loss_vgg_train += errVGG.item()
            
            errVGG.backward()
            self.vgg_optimizer.step()'''
                

            # Display in weights and biases
            
            it_train_counter = len(self.train_dataloader)
            wandb.log({'Iteration': self.epoch * it_train_counter + i, 
                    'Train: Discriminator loss': errD.item(),
                    'Train: Discriminator loss gradient ': gradient_penalty.item(),
                    'Train: Generator loss': errG.item(),
                    #'Train: VGG loss': errVGG.item(),
                    })
            
            if self.show_img_wb:
                self.clean_scan      = clean_scan[0]
                print('------------------ clean_scan: ', self.clean_scan.shape)
                self.noisy_scan      = noisy_scan[0]
                self.predicted_scan  = fake_images[0]
            
            if i%50 == 0:
                save_images_weights_and_biases_dnet('Validation Images', self.results_dir, self.clean_scan, self.noisy_scan, self.predicted_scan, self.epoch)
        return loss_gen_train, loss_disc_train
    
    
    '''def valid_one_epoch(self):
        loss_gen_valid  = 0
        loss_disc_valid = 0
        
        with torch.no_grad():
            self.gen_net.eval()
            self.dis_net.eval()

            for i, (x_1, x_2) in enumerate(self.valid_dataloader):
                fixed  = x_1.to(self.device)
                moving = x_2.to(self.device)
                
                # Generator
                t_0, w_0, t_1, w_1, ls = self.gen_net(fixed, moving)
                _, features_w1      = self.dis_net(w_1) 
                _, features_w0      = self.dis_net(w_0) 
                generator_adv_loss  = self.l2_loss(features_w1, features_w0)

                # Computing the Generator Loss
                registration_affine_loss = self.nn_loss.pearson_correlation(fixed, w_0)
                penalty_affine_loss      = self.energy_loss.energy_loss(t_0)
                registration_deform_loss = self.nn_loss.pearson_correlation(fixed, w_1)
                penalty_deform_loss      = self.energy_loss.energy_loss(t_1)
                loss                     = registration_affine_loss + self.alpha_value * penalty_affine_loss +\
                                           registration_deform_loss + self.beta_value * penalty_deform_loss  +\
                                           self.gamma_value * generator_adv_loss
                loss_gen_valid += loss.item()

                # Discriminator
                real, _ = self.dis_net(w_0)  
                fake, _ = self.dis_net(w_1.detach())
                label_r  = torch.full(real.shape, self.real_label, dtype=torch.float, device=self.device)
                label_f  = torch.full(real.shape, self.fake_label, dtype=torch.float, device=self.device)

                # Computing the Generator Loss
                loss_d_real      = self.disc_loss(real, label_r)
                loss_d_fake      = self.disc_loss(fake, label_f)
                loss_d_v         = (loss_d_real + loss_d_fake) * 0.5
                loss_disc_valid += loss_d_v.item()

                # Display in weights and biases
                it_valid_counter = len(self.valid_dataloader)
                wandb.log({'Iteration': self.epoch * it_valid_counter + i, 
                       'Valid: Similarity Affine loss': registration_affine_loss.item(),
                       'Valid: Penalty Affine loss': self.alpha_value * penalty_affine_loss.item(),
                       'Valid: Similarity Elastic loss': registration_deform_loss.item(),
                       'Valid: Penalty Elastic loss': self.beta_value * penalty_deform_loss.item(),
                       'Valid: Generator Adversarial Loss': generator_adv_loss.item(),
                       'Valid: Total loss': loss.item(),
                       'Valid: Discriminator Loss': loss_d_v.item()
                       })

                if self.show_img_wb:
                    self.clean_scan  = fixed
                    self.moving_draw = moving
                    self.w_0_draw    = w_0
                    self.w_1_draw    = w_1
        
        return loss_gen_valid, loss_disc_valid'''
        
        
    def train(self):
        
        # weights and biases
        wandb.init(project=self.exp_name_wb, entity=self.entity_wb)
        config = wandb.config
        wandb.watch(self.gen_net, log="all")
        early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(self.checkpoints_folder, 'best_model.pth'))
                      
        # Establish convention for real and fake labels during training
        self.real_label      = 1.
        self.fake_label      = 0.
        
        for self.epoch in range(self.start_ep, self.n_epochs):
            # Train
            loss_gen_train, loss_disc_train = self.train_one_epoch()
            #loss_gen_train, loss_disc_train, loss_vgg_train = self.train_one_epoch()
            
            # Test
            #loss_gen_valid, loss_disc_valid = self.valid_one_epoch()
            
            # Save checkpoints
            if self.epoch % 10 == 0:
                name_gen = 'gen_' + str(self.epoch) + '.pth'
                name_dis = 'dis_' + str(self.epoch) + '.pth'
                torch.save(self.gen_net.state_dict(), os.path.join(self.checkpoints_folder, name_gen))
                torch.save(self.dis_net.state_dict(), os.path.join(self.checkpoints_folder, name_dis))
                print('Saving model')
            
            # Visualization of images
            if self.show_img_wb:
                save_images_weights_and_biases_dnet('Validation Images', self.results_dir, self.clean_scan, self.noisy_scan, self.predicted_scan, self.epoch)
                
            # Compute the loss per epoch
            loss_gen_train     /= len(self.train_dataloader)
            loss_disc_train    /= len(self.train_dataloader)
            #loss_vgg_train    /= len(self.train_dataloader)
            
            #loss_gen_valid     /= len(self.valid_dataloader)
            #loss_disc_valid    /= len(self.valid_dataloader)
            
            wandb.log({'epoch': self.epoch,
                    'Train: Total loss by epoch': loss_gen_train,
                    #'Valid: Total loss by epoch': loss_gen_valid,
                    'Train: Discriminator Loss by epoch': loss_disc_train,
                    #'Valid: Discriminator Loss by epoch': loss_disc_valid
                    #'Train: VGG Loss by epoch': loss_vgg_train,
                    })
            
            print("Train epoch : {}/{}, loss_Dis = {:.6f},".format(self.epoch, self.n_epochs, loss_disc_train))
            print("Train epoch : {}/{}, loss_gen = {:.6f},".format(self.epoch, self.n_epochs, loss_gen_train)) 
            #print("Train epoch : {}/{}, loss_VGG = {:.6f},".format(self.epoch, self.n_epochs, loss_vgg_train)) 
            #print("Valid epoch : {}/{}, loss_gen = {:.6f},".format(self.epoch, self.n_epochs, loss_gen_valid))
            #print("Valid epoch : {}/{}, loss_Dis = {:.6f},".format(self.epoch, self.n_epochs, loss_disc_valid))
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            #early_stopping(loss_gen_valid, self.gen_net)
            
            if early_stopping.early_stop:
                print("Early stopping", self.epoch)
                break
            

    def run_train(self):
        # Folders to save outputs
        self.save_outputs()
        
        # Model init
        self.model_init()
        
        # Loss functions
        self.init_loss_functions()
        
        # Optimizers
        self.set_optimizer()
        
        # Dataloader
        self.load_dataloader()
        
        # Train
        self.train()
        