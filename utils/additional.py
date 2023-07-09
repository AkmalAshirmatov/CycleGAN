import numpy as np
from PIL import Image


def tensor2im(input_image):
    """"Converts a Tensor array into a numpy image array"""
    image_tensor = input_image.data
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


def save_image(image_numpy, image_path, h=-1, w =-1):
    """Save a numpy image to the disk"""
    image_pil = Image.fromarray(image_numpy)
    if h != -1:
        image_pil = image_pil.resize((h, w), Image.BICUBIC)
    image_pil.save(image_path)
