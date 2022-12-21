import os
import pickle
import shutil
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from aug_image.aug_image import AugImage
from layer_collection import LayerCollection


class AugE:
    def __init__(self, job_description: dict):
        self.path = job_description['output_path']
        self.lc = LayerCollection()

    def add_data(self, name: str):
        aug_img = AugImage.from_path(self.path, name, bg_component=99)
        self.lc.add_aug_img(aug_img)

    def finalize(self, folder_in="temp", folder_out="out", p_use=1):
        """

        @param folder_in:
        @param folder_out:
        @param p_use:
        @return:
        """
        timestamp = str(time.time())
        out_dir = os.path.join(folder_out, timestamp)
        os.makedirs(out_dir)
        # generate new data
        for fname in os.listdir(folder_in):
            with open(os.path.join(folder_in, fname), 'rb') as f:
                aug_img = pickle.load(f)

            img, labels = aug_img.construct_img()
            plt.imshow(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGRA2RGB))
            plt.show()

            indices = list(range(1, len(aug_img.layers_draw)))
            indices = np.random.choice(indices, size=max(1, round(len(indices) * p_use)), replace=False)
            for idx in indices:
                layer = aug_img.layers_draw[idx]
                substitute = self.lc.get_substitute(layer)

                diff = np.array([a - b for a, b in zip(layer.center_of_mass, substitute.center_of_mass)])
                substitute.pos = layer.pos_orig + diff

                # aug_img.layers_draw.pop(idx)
                aug_img.layers_draw[idx] = substitute

            img, labels = aug_img.construct_img()
            plt.imshow(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGRA2RGB))
            plt.show()
            cv2.imwrite(os.path.join(out_dir, f'{aug_img.name_img}.png'), img)

        # empty temp folder
        for filename in os.listdir(folder_in):
            file_path = os.path.join(folder_in, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    jd = {'output_path': r'C:\tmp\output\2022_09_09_11_37_00'}
    auge = AugE(jd)
    auge.add_data('0000')
    auge.add_data('0001')
    # auge.add_data('0002')
    # auge.add_data('0003')
    auge.finalize()
