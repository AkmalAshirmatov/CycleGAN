import torch
import time
import os

import options.options_train as opt
import data.dataset
import utils.additional
import models.cycle_gan_model


def save_images(images):
    image_dir = 'datasets/generated'
    for label, im_data in images.items():
        im = utils.additional.tensor2im(im_data)
        image_name = f'{label}.png'
        save_path = os.path.join(image_dir, image_name)
        utils.additional.save_image(im, save_path)


if __name__ == '__main__':
    dataset = data.dataset.MyDataset(opt=opt, phase='train')
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True)

    model = models.cycle_gan_model.CycleGANModel(opt)

    total_iters = 0

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            total_iters += opt.batch_size

            model.forward_train(data)

            if total_iters % opt.print_freq == 0: # print training losses and save generated images
                losses = model.get_losses()
                print(losses)
                images = model.get_generated_images()
                save_images(images)

        if epoch % opt.save_epoch_freq == 0: # save model
            print(f'saving the model at the end of epoch {epoch}, iters {total_iters}')
            model.save_networks(epoch)

        model.update_learning_rate() # update learning rates

        print(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time} sec')