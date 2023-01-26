import copy
import os
from datetime import datetime

from tqdm import tqdm
import cv2
import numpy as np

# from .aug_image import AugImage
from data_loader import DataLoader

# corners in order: TOP_LEFT, TOP_RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT

class AugImageFactory:

    def __init__(self, data_loader, min_layer_area=0.0001, seed=1000, nr_data_use=None):
        """
        :param data_loader:
        :param min_layer_area: minimum area of a layer to be regarded in percentage to background area
        """
        np.random.seed(seed)
        self.nr_data_use = nr_data_use
        # self.min_layer_area = min_layer_area
        self.aug_images_orig, self.complete_layers = self.prepare(data_loader)
        # for _, cls in self.complete_layers.items():
        #     for cl in cls:
        #         plt.imshow(cl[0].img_slice)
        #         plt.show()

    def generate(self, n=1, p=0.25, do_aug_rand=False):
        for aug_img in tqdm(self.aug_images_orig):
            yield aug_img
            nr_aug_layers = len(aug_img.layers_draw[1::])
            for _ in range(n):
                aug_img_new = copy.deepcopy(aug_img)
                for layer in aug_img_new.layers_draw:
                    assert np.all(layer.pos == layer.pos_orig)
                p_use = p if p else np.random.rand()
                indices = np.random.choice(list(range(1, nr_aug_layers + 1)), max(1, round(nr_aug_layers * p_use)), replace=False)
                for idx in indices:
                    # for l, l_orig in zip(aug_img_new.layers_draw[1::], aug_img.layers_draw[1::]):
                    l, l_orig = aug_img_new.layers_draw[idx], aug_img.layers_draw[idx]
                    assert np.all(l.pos == l.pos_orig)
                    assert np.all(l_orig.pos == l.pos_orig)
                    if l.component != 1:
                        continue
                    img_nr = int(aug_img.name_img)
                    candidates = self.complete_layers.get(l.component)
                    # candidates = [c[0] for c in candidates]

                    if candidates:
                        candidates = [c[0] for c in candidates if abs(c[1] - img_nr) < 12]
                        # scales = [np.sqrt(np.prod(l.img_slice.shape[0:2]) / np.prod(c.img_slice.shape[0:2])) for c in candidates]
                        scales = [np.sqrt(l.size / c.size) for c in candidates]
                        probs = [1 / (np.sqrt(np.sum(np.square(l.pos - c.pos))) + 1) if scale < 2 else 0 for c, scale in zip(candidates, scales)]

                        sum_probs = np.sum(probs)
                        if sum_probs == 0:
                            continue
                        probs /= sum_probs
                        chosen_idx = np.random.choice(list(range(len(candidates))), p=probs)
                        substitute = copy.deepcopy(candidates[chosen_idx]) # todo copy hier noetig?
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
                        assert np.all(l.pos == l.pos_orig)
                        assert np.all(l_orig.pos == l.pos_orig)
                        # l.img_slice = substitute.img_slice
                        # l.is_complete = substitute.is_complete
                        # l.size = substitute.size
                        # l.center_of_mass = substitute.center_of_mass
                        # l_orig.img_slice = substitute.img_slice
                aug_img_new.sd_augment()
                if do_aug_rand:
                    aug_img_new.perform_targets_random(None,
                                                       rot_range=(0,0),
                                                       shear_range=(0,0),
                                                       tint_range=(-10, 10),
                                                       p_flip=0,
                                                       scale_range=(1, 1),
                                                       do_reset_first=False)
                yield aug_img_new

    def prepare(self, data_loader):
        aug_images_orig = []
        complete_layers = {}
        for count, aug_img in enumerate(data_loader):
            if count == self.nr_data_use:
                break
            aug_images_orig.append(aug_img)
            # img, _ = aug_img.construct_img(with_bboxes_drawn=True)
            # plt.imshow(img)
            # plt.show()
            for layer in aug_img.layers_draw:
                img_nr = int(aug_img.name_img.split('.')[0])
                if layer.is_complete:

                    try:
                        complete_layers[layer.component].append((layer, img_nr))
                    except KeyError:
                        complete_layers[layer.component] = [(layer, img_nr)]
        self.flip_enrichment(aug_images_orig, complete_layers)
        return aug_images_orig, complete_layers

    def flip_enrichment(self, aug_images_orig, complete_layers):
        shape = aug_images_orig[0].img_shape
        for aug_img in aug_images_orig:
            assert np.all(aug_img.img_shape == shape)
        new_layers = {}
        for lc, layers in complete_layers.items():
            for l_orig, img_nr in layers:
                l_new = copy.deepcopy(l_orig)
                l_new.img_slice = np.fliplr(l_new.img_slice)
                l_new.pos[1] = shape[1] - l_new.pos[1]
                l_new.pos_orig[1] = shape[1] - l_new.pos_orig[1]
                l_new.calc_center_of_mass()
                try:
                    new_layers[l_new.component].append((l_new, img_nr))
                except KeyError:
                    new_layers[l_new.component] = [(l_new, img_nr)]
        for lc, layers in new_layers.items():
            complete_layers[lc].extend(layers)

if __name__ == '__main__':

    folder_name = 'factory_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dir_base = os.path.abspath(os.path.join('results', folder_name))
    print(f'will write to: {dir_base}')
    dir_imgs_train = os.path.join(dir_base, 'images', 'train')
    os.makedirs(dir_imgs_train)
    print(f"created: {dir_imgs_train}")
    data_loader = DataLoader(r'C:\Users\HEW\Projekte\syclops-dev\output\2023_01_25_15_01_49', step_exclude=0)
    # data_loader.pickle_dataset_as_aug_images()
    print('start preparing data')
    factory = AugImageFactory(data_loader, seed=42)
    # factory = AugImageFactory(data_loader, seed=42, nr_data_use=38)
    print('done preparing data')
    for i, aug_img in enumerate(factory.generate(n=1, p=0., do_aug_rand=True)):
        # img_orig, labels_orig = aug_img_orig.construct_img()
        data_loader.pickle_aug_img(os.path.join(dir_base, 'AugImages'), aug_img)
        img, labels = aug_img.construct_img()
        # cv2.imwrite(os.path.join(dir_imgs_train, f'{i}_orig.png'), img_orig)
        cv2.imwrite(os.path.join(dir_imgs_train, f'{i}_aug.png'), img)
        # cv2.imwrite(os.path.join(dir_imgs_train, f'{i}_labels.png'), np.asarray(labels * (255 / labels.max()), dtype=np.uint8))
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGRA2RGB)), ax2.imshow(
        #     cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)), plt.show()
