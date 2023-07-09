import random


class ImagePool():
    """Image buffer class

    To reduce model oscillation we update discriminators using a history of
    generated images rather than the ones produced by the latest generators
    (C) from the the article
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class,
        pool_size equals to 50 as in the article"""

        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []

    def query(self, image):
        """In the article buffer stores the 50 previously created images
        If size of buffer is less than 50, add current image
        Else delete first image from the pool and add current

        Parameters:
            image: the latest generated image from the generator

        Return random image from the the pooled images
        """
        self.images.append(image)
        if len(self.images) > self.pool_size:
            self.images = self.images[-self.pool_size:]
        random_id = random.randint(0, len(self.images)-1)
        return self.images[random_id]
