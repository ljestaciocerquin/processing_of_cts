import torch
import torch.nn            as nn
import torch.nn.functional as F
from   collections         import OrderedDict
from   metrics             import calculate_gradient_penalty

def conv_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.Conv2d
    elif dim_conv == 3:
        return nn.Conv3d

def max_pool_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.MaxPool2d
    elif dim_conv == 3:
        return nn.MaxPool3d
    
def up_sample_mode(dim_conv: int):
    if dim_conv   == 2:
        return 'nearest'
    elif dim_conv == 3:
        return 'trilinear'
    
def conv_gl_avg_pool_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.AdaptiveAvgPool2d
    elif dim_conv == 3:
        return nn.AdaptiveAvgPool3d
    
def concat(x1, x2):
    #x1     = self.up(x1)
    diff_z = x2.size()[2] - x1.size()[2]
    diff_y = x2.size()[3] - x1.size()[3]
    diff_x = x2.size()[4] - x1.size()[4]
    x1  = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                    diff_y // 2, diff_y - diff_y // 2,
                    diff_z // 2, diff_z - diff_z // 2])
    x   = torch.cat([x2, x1], dim=1)
    return x


class ConvVGG(nn.Module):
    def __init__(self, in_channels, out_channels, group_dim, dim, use_bias=False): 
        super(ConvVGG, self).__init__()
        self.conv = nn.Sequential(
            conv_layer(dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GroupNorm(num_groups=group_dim, num_channels=out_channels),
            nn.ReLU        (inplace=True),
            conv_layer(dim)(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GroupNorm(num_groups=group_dim, num_channels=out_channels),
            nn.ReLU        (inplace=True),
            max_pool_layer(dim)(kernel_size=2, stride=2)
        )

    def forward(self, x):
        out = self.conv(x)
        return out  

    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_dim, dim, use_bias=False): 
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            conv_layer(dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GroupNorm(num_groups=group_dim, num_channels=out_channels),
            nn.ReLU        (inplace=True),
            conv_layer(dim)(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GroupNorm(num_groups=group_dim,  num_channels=out_channels),
            nn.ReLU        (inplace=True)
        )

    def forward(self, x):
        #print(x.shape)
        out = self.double_conv(x)
        return out  


class Up_Conv(nn.Module):

    def __init__(self, input_ch, output_ch, group_num, dim):
        super(Up_Conv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample (scale_factor=2,        mode=up_sample_mode(dim)),
            nn.Conv3d   (in_channels=input_ch,  out_channels=output_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups =group_num, num_channels=output_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.up(x)
        return out
    
    
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim, use_bias=False):
        super(OutConv, self).__init__()
        self.conv = conv_layer(dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias)
    
    def forward(self, x):
        out = self.conv(x)
        return out
    
   
class DenoiseCT_Net(nn.Module):
    def __init__(self,
                 input_ch  : int = 1,
                 input_dim : int = [512, 512, None],
                 output_ch : int = 1,
                 group_num : int = 8,
                 filters   : object = [16, 32, 64, 128]
                ):
        super(DenoiseCT_Net, self).__init__()
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.output_ch  = output_ch
        self.group_num  = group_num
        self.filters    = filters
        
        # Encoder Block
        # Filters sequence
        seq          = [self.input_ch] + self.filters
        self.convnet = nn.ModuleDict()
        # Convolutions layers
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):  
            self.convnet['encoder_conv_' + str(i)] = DoubleConv(in_channels=j, out_channels=k, group_dim=self.group_num, dim=len(self.input_dim), use_bias=False)
        # Max pooling layers. Considering from 2 because there is no max pooling after the input_ch nor after the last convolution
        for i, _ in enumerate(seq[2:], start=1):
            self.convnet['encoder_max_pool_' + str(i)] = max_pool_layer(len(self.input_dim))(kernel_size=2, stride=2)
        
        # Decoder Block
        # Filters sequence
        seq = list(reversed(self.filters))
        # Up convolution layers
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):
            self.convnet['decoder_up_conv_' + str(i)] = Up_Conv(input_ch=j, output_ch=k, group_num=self.group_num, dim=len(self.input_dim))
        # Convolution layers
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):
            self.convnet['decoder_conv_' + str(i)] = DoubleConv(in_channels=j, out_channels=k, group_dim=self.group_num, dim=len(self.input_dim),  use_bias=False)
        
        # Deformation field and registered image
        self.convnet['output'] = OutConv(in_channels=self.filters[0], out_channels=self.output_ch, dim=len(self.input_dim), use_bias=False)
    
    
    def forward(self, scan: torch.tensor):
        
        #x = torch.cat((fixed, moving), dim=1) 
        e1 = self.convnet['encoder_conv_1'](scan)

        e2 = self.convnet['encoder_max_pool_1'](e1)
        e2 = self.convnet['encoder_conv_2'](e2)

        e3 = self.convnet['encoder_max_pool_2'](e2)
        e3 = self.convnet['encoder_conv_3'](e3)


        # Latent space
        e4 = self.convnet['encoder_max_pool_3'](e3) 
        e4 = self.convnet['encoder_conv_4'](e4)


        d4 = self.convnet['decoder_up_conv_1'](e4)         
        d4 = concat(d4, e3)     
        d4 = self.convnet['decoder_conv_1'](d4)
        
        d3 = self.convnet['decoder_up_conv_2'](d4)
        d3 = concat(d3, e2) 
        d3 = self.convnet['decoder_conv_2'](d3)
        
        d2 = self.convnet['decoder_up_conv_3'](d3)
        d2 = concat(d2, e1)
        d2 = self.convnet['decoder_conv_3'](d2)
        
        
        output = self.convnet['output'](d2) 
        return output 


'''from torchsummary import summary    
model =  DenoiseCT_Net(input_ch= 1,
                 input_dim = [None, None, None],
                 output_ch = 1,
                 group_num = 8,
                 filters=[8,16, 32, 64])
summary = summary(model.to('cuda'), [(1, 128, 128, 20)])'''




class Encoder(nn.Module):
    def __init__(self,
                input_ch  : int = 1,
                input_dim : int = [512, 512, None],
                group_num : int = 8,
                filters   : object = [32, 64, 128] 
                ):
        super(Encoder, self).__init__()
        """
        Inputs:
            - input_dim  : Dimensionality of the input 
            - latent_dim : Dimensionality of the latent space (Z)
            - groups     : Number of groups in the normalization layers
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.group_num  = group_num
        self.filters    = filters
        modules         = OrderedDict()
        
        for layer_i, layer_filters in enumerate(filters, start=1):

            modules['encoder_block_' + str(layer_i)] = nn.Sequential(
                conv_layer(len(input_dim))(
                    in_channels=input_ch, out_channels=layer_filters, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=layer_filters),
                nn.ReLU(inplace=True)
            )
            input_ch = layer_filters
        self.conv_net = nn.Sequential(modules)


    def forward(self, x):
        x = self.conv_net(x)
        return x
    


class Discriminator(nn.Module):
    def __init__(self,
                 input_ch  : int = 1,
                 input_dim : int = [512, 512, None],
                 group_num : int = 8,
                 filters   : object = [32, 64, 128]):
        super(Discriminator, self).__init__()
        """
        Inputs:
            - input_ch   : Number of input channels of the image. For medical images, this parameter usually is 1
            - latent_dim : Dimensionality of the latent space (Z)
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """          
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.group_num  = group_num
        self.filters    = filters
        out_features_last_layer = 1
        
        # Encoder Block
        self.conv_discriminator =  Encoder(input_ch=self.input_ch, input_dim=self.input_dim, group_num=self.group_num, filters=self.filters)
        
        self.linear_discriminator = nn.Sequential(OrderedDict([
            ('disc_gl_avg_pool' , conv_gl_avg_pool_layer(len(self.input_dim))(output_size=1)),
            ('disc_ft_vec_all'  , nn.Flatten()),
            ('disc_last_linear' , nn.Linear(in_features=self.filters[-1], out_features=out_features_last_layer, bias=False)),
            #('disc_last__act_fn', nn.Sigmoid()), 
        ]))
            
    
    def forward(self, x: torch.tensor):
        # Get last convolution and last layer
        x = self.conv_discriminator(x)
        x = self.linear_discriminator(x)
        return x
    

'''from torchsummary import summary    
model =  Discriminator(input_ch= 1,
                 input_dim = [None, None, None],
                 group_num = 8,
                 filters=[32, 64, 128])
summary = summary(model.to('cuda'), [(1, 128, 128, 20)])'''


class VGG4(nn.Module):
    def __init__(self,
                 input_ch  : int = 1,
                 input_dim : int = [512, 512, None],
                 group_num : int = 8,
                 filters   : object = [64, 128],
                 out_fts   : int = 128):
        super(VGG4, self).__init__()
        """
        Inputs:
            - input_ch   : Number of input channels of the image. For medical images, this parameter usually is 1
            - latent_dim : Dimensionality of the latent space (Z)
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """          
        self.input_ch     = input_ch
        self.input_dim    = input_dim
        self.group_num    = group_num
        self.filters      = filters
        self.out_features = out_fts
    
        # Convolutions layers
        self.convnet = nn.ModuleDict()
        seq          = [self.input_ch] + self.filters
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):  
            self.convnet['vgg_conv_' + str(i)] = ConvVGG(in_channels=j, out_channels=k, group_dim=self.group_num, dim=len(self.input_dim), use_bias=False)
        
        self.features = nn.Sequential(OrderedDict([
            ('disc_gl_avg_pool' , conv_gl_avg_pool_layer(len(self.input_dim))(output_size=1)),
            ('disc_ft_vec_all'  , nn.Flatten()),
            ('disc_last_linear' , nn.Linear(in_features=self.filters[-1], out_features=self.out_features, bias=False)),
        ]))
            
    
    def forward(self, x: torch.tensor):
        # Get last convolution and last layer
        x = self.convnet['vgg_conv_1'](x)
        x = self.convnet['vgg_conv_2'](x)
        x = self.features(x)
        return x
    

'''from torchsummary import summary    
model =  VGG4(input_ch= 1,
                 input_dim = [None, None, None],
                 group_num = 8,
                 filters=[64, 128],
                 out_fts= 128)
summary = summary(model.to('cuda'), [(1, 128, 128, 20)])'''