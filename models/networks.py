import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    We keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    """
    def lambda_rule(epoch): #epoch from zero inside scheduler
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def init_weights(net):
    """Initialize network with weights from N(0, 0.02)"""
    def init_func(layer):  # define the initialization function
        classname = layer.__class__.__name__
        if hasattr(layer, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(layer.weight.data, mean=0.0, std=0.02)
            if hasattr(layer, 'bias') and layer.bias is not None:
                init.constant_(layer.bias.data, 0.0)
    print('initialized network with weights from N(0, 0.02)')
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net):
    """Initialize a network and move to available device"""
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net.to(device)
    init_weights(net)
    return net


class GANLoss(nn.Module):
    """Define GANLoss"""

    def __init__(self):
        """Initialize the GANLoss class
        MSE Loss is used - described in 4 - Training details
        real label = 1.0, take_label = 0.0"""
        nn.Module.__init__(self)
        self.real_label = torch.tensor(1.0)
        self.fake_label = torch.tensor(0.0)
        self.loss = nn.MSELoss()

    def __call__(self, prediction, target_is_real):
        """Loss function call

        Parameters:
            prediction (tensor) - Discriminator's output
            target_is_real (bool) - if the ground truth label is for real images or fake images

        Returns:
            loss between Discriminator's output and ground truth
        """
        target_tensor = self.real_label if target_is_real else self.fake_label
        target_tensor = target_tensor.expand_as(prediction)
        target_tensor = target_tensor.to(prediction.device)
        loss = self.loss(prediction, target_tensor)
        return loss


def define_G(input_nc, output_nc):
    """Create Resnet generator and initialize it
    Using 9 blocks for 256x256 images
    """
    net = ResnetGenerator(input_nc, output_nc, n_blocks=9)
    return init_net(net)


def define_D(input_nc):
    """Create PatchGAN discriminator and initialize it"""
    net = NLayerDiscriminator(input_nc)
    return init_net(net)


class ResnetGenerator(nn.Module):
    """Resnet-based generator"""

    def __init__(self, input_nc, output_nc, n_blocks=9):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            n_blocks (int)      -- the number of ResNet blocks
        """
        nn.Module.__init__(self)

        num_ch = 64
        # input_nc x 256x256 -> input_nc x 262x262 -> 64x256x256
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, num_ch, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(num_ch),
                 nn.ReLU(True)]

        # downsample
        for _ in range(2): # 64x256x256 -> 128x128x128 -> 256x64x64
            model += [nn.Conv2d(num_ch, num_ch * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(num_ch * 2),
                      nn.ReLU(True)]
            num_ch *= 2

        # ResNet blocks 256x64x64 -> 256x64x64 -> ... -> 256x64x64
        for _ in range(n_blocks): # doesn't change number of channels
            model += [ResnetBlock(num_ch)]

        # upsample
        for _ in range(2):  # 256x64x64 -> 128x128x128 -> 64x256x256
            model += [nn.ConvTranspose2d(num_ch, num_ch // 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(num_ch // 2),
                      nn.ReLU(True)]
            num_ch //= 2

        # 64x256x256 -> 64x262x262 -> output_nc x 256x256
        model += [nn.ReflectionPad2d(3),
                nn.Conv2d(num_ch, output_nc, kernel_size=7, padding=0),
                nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Forward pass"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim):
        """Initialize the Resnet block - conv block with skip connection"""
        nn.Module.__init__(self)
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        """Construct a convolutional block
        Reflection padding is used to reduce artifacts (7.2, article)
        Instance normalization is used (4th part, article)

        Parameters:
            dim (int) -- the number of channels in the conv layer

        Returns tensor with same number of channels as given
        """
        conv_block = [nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                nn.InstanceNorm2d(dim),
                nn.ReLU(True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward pass"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator
    https://sahiltinky94.medium.com/understanding-patchgan-9f3c8380c207
    nice explanation why it is 70x70 overlapping image patches discriminator
    """

    def __init__(self, input_nc):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int) -- the number of channels in input images
        """
        nn.Module.__init__(self)

        # input_nc x 256x256 -> 64x128x128
        sequence = [nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, True)]
        # 64x128x128 -> 128x64x64
        sequence += [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True)
        ]
        # 128x64x64 -> 256x32x32
        sequence += [
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True)
        ]
        # 256x32x32 -> 512x31x31
        sequence += [
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True)
        ]
        # 512x31x31 -> 1x30x30
        sequence += [nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Forward pass"""
        return self.model(input)
