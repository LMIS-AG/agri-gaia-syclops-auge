# Created by HEW at 18.05.2021
import numpy as np

def get_crop_slices(mask):
    args = np.nonzero(mask)
    x_0, x_1, y_0, y_1 = np.min(args[0]), np.max(args[0]) + 1, np.min(args[1]), np.max(args[1]) + 1
    slice_x = slice(x_0, x_1)
    slice_y = slice(y_0, y_1)
    return slice_x, slice_y


def compose(img_target: np.array, img_slice: np.array, pos_in: tuple, is_center_pos: bool = False, alpha_scale=1):
    """
    perform compositing. Insert img_slice into full_img
    @param img_target: the target image used for insertion
    @param img_slice: the img with alpha channel that is to be inserted
    @param pos_in: position at which to insert the slice
    @param is_center_pos: whether pos denotes the center or the top left point
    @param alpha_scale: scale alpha value when adding
    @return: slices of bounding box of edited area in full_img
    """
    pos = (pos_in[0] - img_slice.shape[0] // 2,
           pos_in[1] - img_slice.shape[1] // 2) if is_center_pos else pos_in

    s_x = slice(np.clip(pos[0], 0, img_target.shape[0]),
                np.clip(pos[0] + img_slice.shape[0], 0, img_target.shape[0]))
    s_y = slice(np.clip(pos[1], 0, img_target.shape[1]),
                np.clip(pos[1] + img_slice.shape[1], 0, img_target.shape[1]))

    start_x = 0 if pos[0] >= 0 else -pos[0]
    start_y = 0 if pos[1] >= 0 else -pos[1]

    # create slice
    img_slice = img_slice[start_x: start_x + s_x.stop - s_x.start,
                start_y: start_y + s_y.stop - s_y.start]

    # add layer to full image
    transparency = 1 - (alpha_scale * img_slice[:, :, 3] / 255 )
    img_target[s_x, s_y, 0] = img_target[s_x, s_y, 0] * transparency + img_slice[:, :, 0] * (1 - transparency)
    img_target[s_x, s_y, 1] = img_target[s_x, s_y, 1] * transparency + img_slice[:, :, 1] * (1 - transparency)
    img_target[s_x, s_y, 2] = img_target[s_x, s_y, 2] * transparency + img_slice[:, :, 2] * (1 - transparency)

    if img_target.shape[2] == 4:
        transparency_target = 1 - (img_target[s_x, s_y, 3] / 255)
        img_target[s_x, s_y, 3] = (1 - (transparency_target * transparency)) * 255

    return s_x, s_y, img_slice
