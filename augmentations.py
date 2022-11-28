import cv2
import numpy as np
from scipy import ndimage


HSV_LIMITS = np.array([179, 255, 255], dtype=np.uint8)


def zoom(img_slice, pos, f_x, f_y):
    img_zoom = ndimage.zoom(img_slice, (f_x, f_y, 1))

    # calc new position
    pos_mid = pos + np.array(img_slice.shape[0:2]) / 2
    pos_new = pos_mid - np.array(img_zoom.shape[0:2]) / 2
    return img_zoom, pos_new

def flip_lr(img_slice):
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

    # img_slice[np.all(img_slice == 0, 2)] = 255
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

def tint(img_slice, color):

    img_slice = cv2.cvtColor(img_slice[:, :, 0:3], cv2.COLOR_BGR2HSV)  # todo lieber gleich in HSV arbeiten?
    non_black = img_slice[:, :, 3] > 0
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
    img_slice[:, :, 0:3] = cv2.cvtColor(img_slice, cv2.COLOR_HSV2BGR)
    # self.layers_draw[target][1][:,:,0:3] = cv2.cvtColor(img_slice,cv2.COLOR_HSV2BGR)
    # self.layers[target][1][non_black, 1::] = np.clip(img_slice[non_black, 1::] + color[1::], 0, 255)