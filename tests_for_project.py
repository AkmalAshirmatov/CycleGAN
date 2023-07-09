import options.options_train as opt
import data.dataset
from util.image_pool import ImagePool
import torch
import torch.nn as nn
import models.networks
from torchsummary import summary

def check_dataset_and_dataloader(phase):
    import data.dataset
    dataset = data.dataset.MyDataset(opt, phase)
    print(f'len A = {len(dataset.A_paths)}')
    print(f'len B = {len(dataset.B_paths)}')
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True)
    print(len(dataloader))
    for i, data in enumerate(dataloader):  # inner loop within one epoch
        print(data)
        if i > 3:
            break


def test_pool():
    import data.dataset
    dataset = data.dataset.MyDataset(opt, 'train')
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True)
    fake_A_pool = ImagePool(opt.pool_size)
    fake_B_pool = ImagePool(opt.pool_size)
    for i, data in enumerate(dataloader):  # inner loop within one epoch
        A = fake_A_pool.query(i)
        B = fake_A_pool.query(i)
        print(A, B)
        if i > 100:
            break


def image_size():
    '''
    x = torch.zeros((1,256,64,64))
    layer = nn.ConvTranspose2d(256, 128,
                            kernel_size=3, stride=2,
                            padding=1, output_padding=1,
                            bias=False)
    x = layer(x)
    print(x.shape)
    '''
    kw = 4
    padw = 1
    x = torch.zeros((1,3,256,256))
    layer = nn.Conv2d(3, 64, kernel_size=kw, stride=2, padding=padw)
    x = layer(x)
    print(x.shape)

    x = torch.zeros((1,256,32,32))
    layer = nn.Conv2d(256, 512, kernel_size=kw, stride=1, padding=padw, bias=False)
    x = layer(x)
    print(x.shape)

    x = torch.zeros((1,512,31,31))
    layer = nn.Conv2d(512, 1, kernel_size=kw, stride=1, padding=padw)
    x = layer(x)
    print(x.shape)


def test_model_discriminator():
    model = models.networks.NLayerDiscriminator(3)
    print(model)
    summary(model, input_size=(3, 256, 256))


def test_model_resnetblock():
    model = models.networks.ResnetBlock(64)
    print(model)
    summary(model, input_size=(64, 256, 256))


def test_model_resnetgenerator():
    model = models.networks.ResnetGenerator(3, 3, 9)
    print(model)
    summary(model, input_size=(3, 256, 256))


if __name__ == '__main__':
    #check_dataset_and_dataloader('train')
    #check_dataset_and_dataloader('test')
    #test_pool()
    #image_size()
    #test_model_discriminator()
    #test_model_resnetblock()
    test_model_resnetgenerator()
