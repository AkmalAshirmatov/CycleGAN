import options.options_test as opt
import data.dataset
import torch
import models.cycle_gan_model
import os
import utils.additional


counter = 0


def save_images(visuals):
    image_dir = 'datasets/generated'
    global counter
    for label, im_data in visuals.items():
        im = utils.additional.tensor2im(im_data)
        image_name = '%s_%s.png' % (counter, label)
        save_path = os.path.join(image_dir, image_name)
        utils.additional.save_image(im, save_path)
    counter += 1


if __name__ == '__main__':
    dataset_test = data.dataset.MyDataset(opt=opt, phase='test')
    dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=opt.batch_size,
            shuffle=False)

    model = models.cycle_gan_model.CycleGANModel(opt)
    for i, data in enumerate(dataset_test):
        model.forward_test(data)
        # save generated images
        images = model.get_generated_images()
        save_images(images)
        if i >= 10:
            break
