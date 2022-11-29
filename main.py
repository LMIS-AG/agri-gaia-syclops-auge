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
        aug_img = AugImage.from_path(self.path, name)
        # img, labels = aug_img.construct_img()
        # plt.imshow(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGRA2RGB))
        # plt.show()
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
        pass

    def generate(self, aug_images, n=1, p=0.25, do_aug_rand=False):
        for aug_img in tqdm(aug_images):
            nr_aug_layers = len(aug_img.layers_draw[1::])
            for _ in range(n):
                aug_img_new = copy.deepcopy(aug_img)
                for layer in aug_img_new.layers_draw:
                    assert np.all(layer.pos == layer.pos_orig)
                p_use = p if p else np.random.rand()
                indices = np.random.choice(list(range(1, nr_aug_layers + 1)), max(1, round(nr_aug_layers * p_use)),
                                           replace=False)
                for idx in indices:
                    # for l, l_orig in zip(aug_img_new.layers_draw[1::], aug_img.layers_draw[1::]):
                    l, l_orig = aug_img_new.layers_draw[idx], aug_img.layers_draw[idx]
                    assert np.all(l.pos == l.pos_orig)
                    assert np.all(l_orig.pos == l.pos_orig)
                    if l.component != 1:
                        continue
                    img_nr = int(aug_img.name_img)
                    candidates = self.complete_layers.get(l.component)
                    candidates = [c[0] for c in candidates if abs(c[1] - img_nr) < 12]
                    # candidates = [c[0] for c in candidates]

                    if candidates:
                        # scales = [np.sqrt(np.prod(l.img_slice.shape[0:2]) / np.prod(c.img_slice.shape[0:2])) for c in candidates]
                        scales = [np.sqrt(l.size / c.size) for c in candidates]
                        probs = [1 / (np.sqrt(np.sum(np.square(l.pos - c.pos))) + 1) if scale < 2 else 0 for c, scale in
                                 zip(candidates, scales)]

                        sum_probs = np.sum(probs)
                        if sum_probs == 0:
                            continue
                        probs /= sum_probs
                        chosen_idx = np.random.choice(list(range(len(candidates))), p=probs)
                        substitute = copy.deepcopy(candidates[chosen_idx])  # todo copy hier noetig?
                        substitute.depth = l.depth
                        aug_img_new.layers_draw[idx] = substitute
                        if do_aug_rand:
                            scale = scales[chosen_idx]
                            scale_range = (scale * 0.9, scale * 1.1)
                            # scale_range = (scale, scale)
                            targets = np.array([idx])
                            rot_range = (-5, 5)
                            shear_range = (-0.3, 0.3)
                            tint_range = (0, 0)
                            p_flip = 0.
                            aug_img_new.perform_targets_random(targets,
                                                               rot_range=rot_range,
                                                               shear_range=shear_range,
                                                               tint_range=tint_range,
                                                               p_flip=p_flip,
                                                               scale_range=scale_range,
                                                               do_reset_first=False)

                        diff = np.array([a - b for a, b in zip(l.center_of_mass, substitute.center_of_mass)])
                        substitute.pos = l.pos_orig + diff


if __name__ == '__main__':
    jd = {'output_path': r'C:\tmp\output\2022_09_09_11_37_00'}
    auge = AugE(jd)
    auge.add_data('0000')
    auge.add_data('0001')
    # auge.add_data('0002')
    # auge.add_data('0003')
    auge.finalize()
