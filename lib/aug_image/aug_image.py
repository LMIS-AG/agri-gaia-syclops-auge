from __future__ import annotations
import logging
import copy
import os
from os.path import join

import PIL
from numpy.random import choice
from scipy import ndimage
from matplotlib import pyplot as plt

try:
    from .aug_image_base import AugImageBase
    from.utils import encode_pil_to_base64, decode_base64_to_image
except ModuleNotFoundError:
    from aug_image.aug_image_base import AugImageBase
    from aug_image.utils import encode_pil_to_base64, decode_base64_to_image

import cv2
import numpy as np
import time

HSV_LIMITS = np.array([179, 255, 255], dtype=np.uint8)


class AugImage(AugImageBase):

    @classmethod
    def from_path(cls, step_num, bg_component,
                  path_bgr, path_depth, path_instance, path_component, **kwargs) -> AugImage:

        files_found = [os.path.isfile(x) for x in (path_bgr, path_depth, path_instance, path_component)]
        assert all(files_found), 'files missing'

        img = cv2.imread(path_bgr)
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        instances_map = np.load(path_instance)['array']
        components_map = np.load(path_component)['array']
        depths_map = np.load(path_depth)['array']
        return cls.from_data(img_bgra, instances_map, components_map, depths_map, str(step_num), bg_component, **kwargs)

    @classmethod
    def from_data(cls, img: np.array, instances_map: np.array, components_map: np.array, depths_map: np.array,
                  name_img: str, bg_component: int, min_layer_area: float = 0.0002,
                  cut_off_value: int = 40, use_sd:bool = False) -> AugImage:
        layers = cls.create_layers(img, components_map, instances_map, depths_map, bg_component, min_layer_area, use_sd)
        return cls(layers, cut_off_value, name_img, img.shape, use_sd)

    def reset(self):
        for target in np.arange(self.nr_layers):
            self.layers_draw[target] = copy.deepcopy(self.layers_orig[target])
            self.layers_draw[target] = copy.deepcopy(self.layers_orig[target])


    def sd_augment(self):
        import requests as requests
        def make_grid(self, images_rgb, margin_full=200):
            assert margin_full % 2 == 0
            margin = margin_full // 2

            def split_at_n(widths, heights, n):
                borders = np.cumsum(widths)
                target_width = borders[-1] / (n + 1)
                split_indices = [np.argmin(np.abs(borders - target_width * i)) + 1 for i in range(1, n + 1)]
                grid_widths = []
                grid_heights = []
                for i, si in enumerate(split_indices):
                    start = split_indices[i - 1] if i > 0 else 0
                    grid_heights.append(np.max(heights[start: si]))
                    grid_widths.append(np.sum(widths[start: si]))
                grid_widths.append(np.sum(widths[split_indices[-1]::]))
                grid_heights.append(np.max(heights[split_indices[-1]::]))
                return np.max(grid_widths), grid_heights, split_indices

            widths = [img.shape[1] + margin_full for img in images_rgb]
            heights = [img.shape[0] + margin_full for img in images_rgb]
            width, grid_heights, split_indices = split_at_n(widths, heights, n=int(np.sqrt(len(images_rgb))) - 1)
            rows = []
            # for i, (si, h) in enumerate(zip(split_indices, grid_heights[0:-1])):
            grid = np.zeros((np.sum(grid_heights), width, 3), dtype=np.uint8)
            slices = []
            last_h = 0
            for i, h in enumerate(grid_heights):
                # row = np.zeros((h, width, 3), dtype=np.uint8)
                start = split_indices[i - 1] if i > 0 else 0
                stop = split_indices[i] if i < len(split_indices) else len(images_rgb)
                last_w = 0
                for img in images_rgb[start: stop]:
                    slice_h = slice(last_h + margin, img.shape[0] + last_h + margin)
                    slice_w = slice(last_w + margin, img.shape[1] + last_w + margin)
                    grid[slice_h, slice_w] = img
                    last_w += img.shape[1] + margin_full
                    slices.append((slice_h, slice_w))
                last_h += h
            return grid, slices

        layer_grid = []
        images_grid = []
        grid_size = 9

        layers, images = [], []
        for i, l in enumerate(self.layers_draw):
            img = l.img_slice
            if img.shape[0] * img.shape[1] < 10000 or l.component != 2:
                continue
            layers.append(l)
            mask = img[:,:,3] == 0
            img[mask] *=0
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            images.append(img_rgb)
            if len(layers) == grid_size:
                layer_grid.append(layers)
                images_grid.append(images)
                layers, images = [], []
                # cv2.imshow('sdf', img_rgb)
                # cv2.waitKey()

        for images, layers in zip(images_grid, layer_grid):
            grid, slices = self.make_grid(images)
            image = PIL.Image.fromarray(grid)
            image_enc = encode_pil_to_base64(image)
            # images.append(image_enc)

            payload = {'init_images': [image_enc],
                       'prompt': '9 plants in a grid on black background. '
                                 'plants are very detailed and have imperfections. 4k.',
                       'negative_prompt': 'low resolution. pixelated.',
                       'denoising_strength': 0.4,
                       'cfg_scale': 7,
                       'batch_size': 1}
            response = requests.post(url='http://127.0.0.1:7860/sdapi/v1/img2img', json=payload)
            image_response = decode_base64_to_image(response.json()['images'][0])
            image_response_rgb = np.array(image_response)
            image_response_hsv = cv2.cvtColor(image_response_rgb, cv2.COLOR_RGB2HSV)
            image_response = cv2.cvtColor(image_response_rgb, cv2.COLOR_RGB2BGRA)
            image_response[:,:,3][image_response_hsv[:,:,2] < 50] = 0
            image_response = cv2.resize(image_response, (grid.shape[1], grid.shape[0]))
            plt.imshow(grid)
            plt.show()
            plt.imshow(image_response_rgb)
            plt.show()
            for i, l in enumerate(layers):
                img_slice = image_response[slices[i]]
                l.img_slice = img_slice

        # cv2.imshow('sdf', image_response)
        # cv2.waitKey()

        # for i, l in enumerate(layers_used):
        #     image_response = decode_base64_to_image(response.json()['images'][i])
        #     image_response = cv2.cvtColor(np.array(image_response), cv2.COLOR_RGB2BGRA)
        #     image_response = cv2.resize(image_response, (img.shape[1], img.shape[0]))
        #     image_response[:,:,3][np.sum(image_response[:,:,0:3], -1) < 100] = 0
        #     # cv2.imshow('sdf', image_response)
        #     # cv2.waitKey()
        #     l.img_slice = image_response

    def perform_targets_random(self,
                               targets=None,
                               scale_range=(0.8, 1.2),
                               rot_range=(-9, 9),
                               shear_range=(-0.3, 0.3),
                               tint_range=(-10, 10),
                               p_flip=0.,
                               do_reset_first=True):
        '''
        if no targets given apply on all targets (except background)
        :param targets:
        :param scale_range:
        :param rot_range:
        :param tint_range:
        :return:
        '''

        if targets is None:
            targets = np.arange(self.nr_layers)
            # np.random.shuffle(targets)

        rot_degrees, scale_factors, shear_factors, tint_colors, flips_lr = self.generate_rand_params(targets, rot_range,
                                                                                                     scale_range,
                                                                                                     shear_range,
                                                                                                     tint_range, p_flip)

        for i, t in enumerate(targets):
            self.perform_augmentation(t, scale_factors[i], rot_degrees[i], tint_colors[i], shear_factors[i],
                                      flips_lr[i], do_reset_first)

    def generate_rand_params(self, targets, rot_range, scale_range, shear_range, tint_range, p_flip):
        incomplete_layers = [i for i in targets if not self.layers_draw[i].is_complete]
        scale_factors = np.random.uniform(scale_range[0], scale_range[1], targets.size)
        # scale_factors[0] = max(1, scale_factors[0])
        scale_factors[incomplete_layers] = 1
        # scale_factors[1] = 1 # Horizont todo remove?
        shear_factors = np.random.uniform(shear_range[0], shear_range[1], (targets.size, 2))
        shear_factors[incomplete_layers] *= 0
        # shear_factors[1] *= 0 # Horizont todo remove?
        # rot_degrees = [np.random.uniform(rot_range[0] * (sf - scale_range[0]) / (scale_range[1] - scale_range[0]),
        #                                  rot_range[1] * (sf - scale_range[0]) / (scale_range[1] - scale_range[0]))
        #                for sf in scale_factors]  # limit rot range based on scale
        rot_degrees = np.random.uniform(rot_range[0], rot_range[1], targets.size)

        rot_degrees[incomplete_layers] = 0
        # rot_degrees[1] = 0 # Horizont todo remove?
        tint_colors = np.random.uniform(tint_range[0], tint_range[1], (targets.size, 3))
        do_flips = np.random.choice([True, False], targets.size, p=[p_flip, 1 - p_flip])
        do_flips[incomplete_layers] = False
        return rot_degrees, scale_factors, shear_factors, tint_colors, do_flips

    def perform_augmentation(self, target, scale_factor=1, rot_degree=0, tint_color=0, shear_factors=None,
                             do_flip=False, do_reset_first=True):
        '''

        :param target:
        :param scale_factor:
        :param rot_degree:
        :param tint_color:
        :return:
        '''
        # reset drawing layer
        if do_reset_first:
            self.layers_draw[target] = copy.deepcopy(self.layers_orig[target])

        # perform augmentations
        if scale_factor != 1.:
            self.change_size(target, scale_factor)
        if shear_factors is not None:
            self.shear(target, shear_factors[0], shear_factors[1])
        if rot_degree != 0.:
            self.rotate(target, rot_degree)

        tint_color = tint_color if isinstance(tint_color, np.ndarray) else np.array((tint_color, 0, 0))
        if not np.all(tint_color == 0):
            self.tint(target, tint_color)

        if do_flip:
            self.flip_lr(target)

        img_slice = self.layers_draw[target].img_slice  # pos?
        cut_off_mask = img_slice[:, :, 3] < self.cut_off_value
        img_slice[cut_off_mask, 3] *= 0
        img_slice[np.bitwise_not(cut_off_mask), 3] = 255

    def zoom(self, target, f_x, f_y):
        pos, img_slice = self.layers_draw[target].pos, self.layers_draw[target].img_slice
        img_zoom = ndimage.zoom(img_slice, (f_x, f_y, 1))

        # calc new position
        pos_mid = pos + np.array(img_slice.shape[0:2]) / 2
        pos = pos_mid - np.array(img_zoom.shape[0:2]) / 2

        # set new layer
        self.layers_draw[target].img_slice = img_zoom
        self.layers_draw[target].pos = pos
        self.layers_draw[target].calc_center_of_mass()

    def flip_lr(self, target):
        img_slice = self.layers_draw[target].img_slice
        flipped = np.fliplr(img_slice)
        self.layers_draw[target].img_slice = flipped
        self.layers_draw[target].calc_center_of_mass()

    def rotate(self, target: int, degree: float):
        img_slice = self.layers_draw[target].img_slice
        rotated = ndimage.rotate(img_slice, degree)
        self.layers_draw[target].img_slice = rotated
        self.layers_draw[target].calc_center_of_mass()

    def shear(self, target: int, degree_h: float, degree_v: float):
        pos, img_slice = self.layers_draw[target].pos, self.layers_draw[target].img_slice

        height, width, colors = img_slice.shape

        transform = np.array([[1, degree_v, 0],
                              [degree_h, 1, 0],
                              [0, 0, 1]])

        corners = [[0, 0, 0], [0, width, 0], [height, 0, 0], [height, width, 0]]
        corners_trans = [np.dot(transform, c) for c in corners]
        corners_trans_x = [c[0] for c in corners_trans]
        corners_trans_y = [c[1] for c in corners_trans]

        c_h = max(corners_trans_y) - min(corners_trans_y)
        c_v = max(corners_trans_x) - min(corners_trans_x)

        height_out, width_out = int(np.ceil(c_v)), int(np.ceil(c_h))
        sheared_array = ndimage.affine_transform(img_slice,
                                                 np.linalg.inv(transform),
                                                 offset=(min(corners_trans_x) * height / height_out,
                                                         min(corners_trans_y) * width / width_out,
                                                         0),
                                                 output_shape=(height_out, width_out, colors))

        self.layers_draw[target].img_slice = sheared_array
        self.layers_draw[target].calc_center_of_mass()

    def change_size(self, target: int, factor: float):
        pos, img_slice = self.layers_draw[target].pos, self.layers_draw[target].img_slice
        mode = cv2.INTER_CUBIC
        if target == 0:
            target_shape = (img_slice.shape[1], img_slice.shape[0])
            w, h = round(target_shape[0] / factor), round(target_shape[1] / factor)
            diff0, diff1 = (target_shape[0] - w) / 2, (target_shape[1] - h) / 2
            idx0_w, idx1_w = int(diff0), int(diff0 + 0.5)
            idx0_h, idx1_h = int(diff1), int(diff1 + 0.5)
            img_slice = img_slice[idx0_h: -idx1_h, idx0_w: -idx1_w]
            img_new = cv2.resize(img_slice, target_shape, mode)
        else:
            # create new rescaled image
            if factor >= 1:
                target_shape = (max(int(img_slice.shape[1] * factor), img_slice.shape[1] + 1),
                                max(int(img_slice.shape[0] * factor), img_slice.shape[0] + 1))
            else:
                target_shape = (min(max(1, int(img_slice.shape[1] * factor)), max(1, img_slice.shape[1] - 1)),
                                min(max(1, int(img_slice.shape[0] * factor)), max(1, img_slice.shape[0] - 1)))
                mode = cv2.INTER_AREA

            img_new = cv2.resize(img_slice, target_shape, mode)  # , cv2.INTER_LANCZOS4)
            # img_new[img_new[:, :, 2] < self.cut_off_value] *= 0  # set almost black pixels to all 0 todo:bettersolution?

            # calc new position
            pos_mid = pos + np.array(img_slice.shape[0:2]) / 2
            pos = pos_mid - np.array(img_new.shape[0:2]) / 2

        # set new layer
        self.layers_draw[target].img_slice = img_new
        self.layers_draw[target].pos = pos
        self.layers_draw[target].calc_center_of_mass()

    def tint(self, target, color):

        img_slice_bgra = self.layers_draw[target].img_slice
        img_slice = cv2.cvtColor(img_slice_bgra[:, :, 0:3], cv2.COLOR_BGR2HSV)  # todo lieber gleich in HSV arbeiten?
        non_black = img_slice_bgra[:, :, 3] > 0
        for i, c in enumerate(color):
            if c < 0:
                c_uint = np.uint8(abs(c))
                if i != 0:
                    mask = img_slice[:, :, i] >= c_uint
                    img_slice[np.bitwise_and(non_black, mask), i] -= c_uint
                    img_slice[np.bitwise_and(non_black, np.bitwise_not(mask)), i] = 0
                else:
                    img_slice[non_black, 0] -= c_uint
                    img_slice[non_black, 0] = img_slice[non_black, 0] % (HSV_LIMITS[0] + 1)
            else:
                c_uint = np.uint8(c)
                if i != 0:  # if not h value do not cycle
                    mask = img_slice[:, :, i] <= HSV_LIMITS[i] - c_uint
                    img_slice[np.bitwise_and(non_black, mask), i] += c_uint
                    img_slice[np.bitwise_and(non_black, np.bitwise_not(mask)), i] = HSV_LIMITS[i]
                else:
                    img_slice[non_black, 0] += c_uint
                    img_slice[non_black, 0] = img_slice[non_black, 0] % (HSV_LIMITS[0] + 1)
        img_slice_bgra[:, :, 0:3] = cv2.cvtColor(img_slice, cv2.COLOR_HSV2BGR)

    def plot(self, info='', save_path='', do_show=True):
        from matplotlib import pyplot as plt
        img, img_labels = self.construct_img()
        fig, axs, plots = self.plot_data
        if fig is None:
            fig, axs = plt.subplots(1, 2)
            plot_img = axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
            axs[0].set_title('augmented image')
            axs[1].set_title('labels')
            plot_labels = axs[1].imshow(img_labels)
            plots = [plot_img, plot_labels]
            self.plot_data = (fig, axs, plots)
        else:
            plots[0].set_data(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
            plots[1].set_data(img_labels)
            if do_show:
                fig.canvas.draw_idle()
        plt.suptitle(info)
        if do_show:
            plt.waitforbuttonpress()
        if save_path:
            plt.savefig(os.path.join(save_path, str(time.time()) + '.png'), dpi=200)
