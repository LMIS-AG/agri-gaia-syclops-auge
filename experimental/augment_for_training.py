import os

import cv2
import numpy as np


def make_grid(images, margin_full=200):
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

    widths = [img.shape[1] + margin_full for img in images]
    heights = [img.shape[0] + margin_full for img in images]
    width, grid_heights, split_indices = split_at_n(widths, heights, n=max(1, int(np.sqrt(len(images))) - 1))
    # for i, (si, h) in enumerate(zip(split_indices, grid_heights[0:-1])):
    grid = np.zeros((np.sum(grid_heights), width, images[0].shape[2]), dtype=np.uint8)
    slices = []
    last_h = 0
    for i, h in enumerate(grid_heights):
        # row = np.zeros((h, width, 3), dtype=np.uint8)
        start = split_indices[i - 1] if i > 0 else 0
        stop = split_indices[i] if i < len(split_indices) else len(images)
        last_w = 0
        for img in images[start: stop]:
            slice_h = slice(last_h + margin, img.shape[0] + last_h + margin)
            slice_w = slice(last_w + margin, img.shape[1] + last_w + margin)
            grid[slice_h, slice_w] = img
            last_w += img.shape[1] + margin_full
            slices.append((slice_h, slice_w))
        last_h += h
    return grid, slices

def resize(img, px, py):
    width = int(img.shape[1] * px)
    height = int(img.shape[0] * py)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

if __name__ == '__main__':
    path = r'C:\Users\HEW\Projekte\plant_disease_datasets\corn\data\Common_Rust'
    files = [x for x in os.listdir(path) if x.endswith('.JPG')]
    for i in range(100):
        chosen = np.random.choice(files, np.random.randint(1, 10))
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(path, f)), cv2.COLOR_BGR2RGB) for f in chosen]
        ps = np.random.rand(len(imgs), 2) + 0.5
        imgs_resize = [resize(img, px, py) for img, (px, py) in zip(imgs, ps)]
        grid, _ = make_grid(imgs_resize, 50) if len(chosen) > 1 else (imgs[0], None)
        cv2.imwrite(f'out/{i}.jpg', grid)

        with open(f'out/{i}' + '.txt', 'w') as f:
            s_p = 'leaf' if len(chosen) == 1 else 'leafs'
            f.write(f'closeup of {len(chosen)} {s_p} with common rust disease on black background')
