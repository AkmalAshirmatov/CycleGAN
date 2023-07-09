import os
import torchvision.transforms as transforms


def make_dataset(directory):
    images = []
    assert os.path.isdir(directory)
    for root, _, fnames in sorted(os.walk(directory)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images


def get_transform(opt, method=transforms.InterpolationMode.BICUBIC):
    transform_list = []
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))

    transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)
