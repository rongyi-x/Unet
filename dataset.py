#codeing=utf-8
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image


class Carvana(Dataset):

    def __init__(self, image_dir: str, mask_dir: str, transform):
        super(Carvana, self).__init__()

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = os.listdir(self.image_dir)
        self.mask_images = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.images[item])
        mask_path = os.path.join(self.mask_dir, self.mask_images[item])

        # print(image_path)
        # print(mask_path)

        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        # print(image.shape, mask.shape)

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
            # 将(0, 255) -> (0, 1)
            mask[mask > 0] = 1.0

        return image, mask

# # test
# from config import get_config
# config = get_config()
# my_dataset = Carvana(config['train_images'], config['train_masks'])
#
# for _, image in enumerate(my_dataset):
#     pass

