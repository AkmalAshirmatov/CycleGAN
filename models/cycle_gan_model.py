import torch
import itertools
from utils.image_pool import ImagePool
from . import networks
from collections import OrderedDict
import os


class CycleGANModel():
    """
    This class implements the CycleGAN model

    Using ResNet generator with 9 blocks as described in the paper
    And PatchGAN 70x70 discriminator
    """

    def __init__(self, opt):
        """Initialize the CycleGAN class"""
        self.opt = opt
        self.isTrain = opt.isTrain
        self.optimizers = []
        self.schedulers = []
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            self.visual_names.append('idt_A')
            self.visual_names.append('idt_B')
        # if train, need Generators and Discriminators
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else: # for test need only Generators
            self.model_names = ['G_A', 'G_B']

        # define networks
        # G_A: domain A -> domain B, G_B: domain B -> domain A
        # D_A - discriminates images in domain B from real and fakes, D_B - in domain A
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc)
        if self.isTrain:
            # define discriminators
            self.netD_A = networks.define_D(opt.output_nc,)
            self.netD_B = networks.define_D(opt.input_nc)
            # create ImagePool for previously generated images for both domains
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss().to(self.device)  # define discriminator loss
            self.criterionCycle = torch.nn.L1Loss() # define cycle loss
            self.criterionIdt = torch.nn.L1Loss() # define cycle loss
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        else:
            # load networks and turn on eval mode
            self.load_networks(opt.epoch)
            for name in self.model_names:
                net = getattr(self, 'net' + name)
                net.eval()
        self.print_networks(opt.verbose)

    def backward_D_basic(self, netD, real, fake):
        """Calculate adversarial loss for the discriminator"""
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combine loss and calculate gradients
        # Here is multiplied by 0.5, because in Apppendix 7.1 there is written:
        # We divide the objective (loss) by 2 while optimizing D, which slows down
        # the rate at which D learns, relative to the rate of G
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A in domain B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B in domain A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A_B = self.opt.lambda_A_B
        # Identity loss, was needed for Monet's paintings to test
        # In the article: "For example, the generator often maps paintings of daytime to
        # photographs taken during sunset without Identity loss
        if lambda_idt > 0:
            # In the Appendix 7.1 they use 0.5*lambda_A_B for weight of Identity loss
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_A_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A_B * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        # Adversarial loss. Give discriminator fake images from B and calculate loss for generator A
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # Adversarial loss. Give discriminator fake images from A and calculate loss for generator B
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Cycle consistency loss for A domain
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A_B
        # Cycle consistency loss for B domain
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_A_B
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()

    def forward_train(self, input):
        """For pass in train mode"""
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        # optimize generators
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # don't update discriminators' weights
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights

    def forward_test(self, input):
        """For pass in test mode"""
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)

        with torch.no_grad():
            self.fake_B = self.netG_A(self.real_A)  # G_A(A)
            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def get_generated_images(self):
        """Return generated images"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_losses(self):
        """Return traning losses"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    # almost copy https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'learning rate {old_lr:.7f} -> {lr:.7f}')

    # almost copy https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
    def save_networks(self, epoch):
        """Save all the networks to the disk with file name {epoch}_net_{name}.pth"""
        for name in self.model_names:
            save_filename = f'{epoch}_net_{name}.pth'
            save_path = os.path.join(self.opt.save_dir, save_filename)
            net = getattr(self, 'net' + name)
            torch.save(net.cpu().state_dict(), save_path)
            net.to(self.device)

    # almost copy https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
    def load_networks(self, epoch):
        """Load all the networks from the disk with file name {epoch}_net_{name}.pth"""
        for name in self.model_names:
            load_filename = f'{epoch}_net_{name}.pth'
            load_path = os.path.join(self.opt.save_dir, load_filename)
            net = getattr(self, 'net' + name)
            print(f'loading the model from {load_path}')
            state_dict = torch.load(load_path, map_location=str(self.device))
            net.load_state_dict(state_dict)

    # almost copy https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture"""
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f'[Network {name}] Total number of parameters : {num_params/1e6:.3f} M')
        print('-----------------------------------------------')

    # almost copy https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/cycle_gan_model.py
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False/True for nets
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks requires gradients or not
        """
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
