import cv2
import numpy as np
import os

import scipy.ndimage
from matplotlib import pyplot as plt
from data_loader import DataLoader
from aug_image.utils import compose

particle_in = cv2.imread(r'C:\Users\HEW\Projekte\syndata_augmentation\data\dirt.png', cv2.IMREAD_UNCHANGED)
n = 50000
n_scale, n_bright, n_blur, n_rot = 10, 10, 3, 5
particles = np.empty((n_scale, n_bright, n_blur, n_rot), dtype=object)
scales = np.linspace(0.05, 0.005, n_scale)
rots = np.linspace(0, 360, n_rot)
blurs = np.linspace(1, 31, n_blur)
brights = np.linspace(10, 250, n_bright)
for i, scale in enumerate(scales):
    for j, bright in enumerate(brights):
        for k, blur in enumerate(blurs):
            for l, rot in enumerate(rots):
                'size'
                p = cv2.resize(particle_in, None, fx=scale, fy=scale)

                'brightness'
                alpha = p[:, :, 3]
                p = cv2.cvtColor(p, cv2.COLOR_BGR2HSV)
                v_mean = np.mean(p[:, :, 2])
                if v_mean == 0:
                    p[:, :, 2] = bright
                else:
                    v_new = p[:, :, 2] * (bright / v_mean)
                    v_new = np.array(np.clip(v_new, 0, 255), dtype=np.uint8)
                    p[:, :, 2] = v_new
                p = cv2.cvtColor(p, cv2.cv2.COLOR_HSV2BGR)
                p = cv2.cvtColor(p, cv2.cv2.COLOR_BGR2BGRA)
                p[:, :, 3] = alpha

                'blur'
                blur = int(blur)
                kernel_motion_blur = np.zeros((blur, blur))
                kernel_motion_blur[int((blur - 1) / 2), :] = np.ones(blur)
                kernel_motion_blur = kernel_motion_blur / blur
                p = cv2.filter2D(p, -1, kernel_motion_blur)

                'rotate'
                p = scipy.ndimage.rotate(p, rot)

                particles[i, j, k, l] = p

data_loader = DataLoader(r'C:\Users\HEW\Projekte\syndata_augmentation\data\GIL Augmentation\2022_10_17_17_13_43',
                         step_exclude=0)
for bgr, depth, instance, component, name_img in data_loader.generate_data():
    depth[depth == 0] = np.max(depth) // 100  # todo ad hoc weil Horizont Entfernung 0
    noise = np.zeros((bgr.shape[0], bgr.shape[1], 4))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    depth_flat = depth.flatten()
    positions = np.random.choice(np.arange(depth_flat.size), n, p=depth_flat / np.sum(depth))

    depth_threshs = np.linspace(0, np.max(depth_flat), n_scale)
    for p in positions:
        coord = np.unravel_index(p, depth.shape)
        # choose depth
        depth_place = np.random.rand() * depth_flat[p]
        i_depth = np.argmin(np.square(depth_threshs - depth_place))

        # choose birightness
        i_bright = np.argmin(np.square(brights - hsv[coord][2]))

        particle = np.random.choice(particles[i_depth, i_bright].flat)
        compose(noise, particle, coord, is_center_pos=True, alpha_scale=0.25)

    plt.imshow(cv2.cvtColor(np.array(noise, dtype=np.uint8), cv2.COLOR_BGRA2RGB))
    plt.show()
    compose(bgr, noise, (0, 0))
    plt.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    plt.show()
