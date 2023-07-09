import os
import random
from PIL import Image

from torch.utils.data import Dataset
from .helping_functions import make_dataset, get_transform


class MyDataset(Dataset):
    """Dataset for CycleGAN"""

    def __init__(self, opt, phase):
        """Initialize the dataset class

        Parameters:
            opt -- experiment options
            phase -- ['train', 'test']
        """
        Dataset.__init__(self)
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, phase + 'A')  # prepate path for images A - /datasets/phaseA
        self.dir_B = os.path.join(opt.dataroot, phase + 'B')  # prepate path for images B - /datasets/phaseB
        self.A_paths = sorted(make_dataset(self.dir_A)) # load images from '/datasets/phaseA'
        self.B_paths = sorted(make_dataset(self.dir_B)) # load images from '/datasets/phaseB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)

    def __getitem__(self, index_A):
        """
        Returns a pair of images (A_image, B_image) 
        from A and B domains as a dictionary
        From domain A it returns image with index=index_A
        From domain B randomly choose pair for A
        """
        A_path = self.A_paths[index_A]
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        return {'A': A, 'B': B}

    def __len__(self):
        """Return len of A domain"""
        return self.A_size