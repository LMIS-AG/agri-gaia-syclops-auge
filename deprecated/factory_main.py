import copy
import os
from datetime import datetime

import cv2

# from .aug_image import AugImage
from data_loader import DataLoader
from aug_image.aug_image_factory import AugImageFactory

folder_name = 'factory_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
dir_base = os.path.abspath(os.path.join('results', folder_name))
print(f'will write to: {dir_base}')
dir_imgs_train = os.path.join(dir_base, 'images', 'train')
os.makedirs(dir_imgs_train)
print(f"created: {dir_imgs_train}")
# data_loader = DataLoader(r'C:\Users\HEW\Projekte\syclops-dev\output\2023_01_25_15_01_49', step_exclude=0)
data_loader = DataLoader(r'C:\Users\HEW\Projekte\syndata_augmentation\data\GIL Augmentation\2022_10_17_17_13_43_last_10', step_exclude=0)
# data_loader.pickle_dataset_as_aug_images()
print('start preparing data')
factory = AugImageFactory(data_loader, seed=42)
# factory = AugImageFactory(data_loader, seed=42, nr_data_use=38)
print('done preparing data')
for i, aug_img in enumerate(factory.generate(n=1, p=0., do_aug_rand=False)):
    # img_orig, labels_orig = aug_img_orig.construct_img()
    # data_loader.pickle_aug_img(os.path.join(dir_base, 'AugImages'), aug_img)
    img, _,_ = aug_img.construct_img()
    # cv2.imwrite(os.path.join(dir_imgs_train, f'{i}_orig.png'), img_orig)
    cv2.imwrite(os.path.join(dir_imgs_train, f'{i}_aug.png'), img)
    # cv2.imwrite(os.path.join(dir_imgs_train, f'{i}_labels.png'), np.asarray(labels * (255 / labels.max()), dtype=np.uint8))
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGRA2RGB)), ax2.imshow(
    #     cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)), plt.show()