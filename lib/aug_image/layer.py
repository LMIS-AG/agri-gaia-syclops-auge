import numpy as np
import scipy

class Layer:
    def __init__(self, pos: np.array, img_slice: np.array, component: int, depth: float):
        self.pos = pos
        self.pos_orig = np.copy(pos)
        self.img_slice = img_slice
        self.component = component
        self.depth = depth
        self.size = np.sum(img_slice[:, :, 3] > 0)
        self.calc_center_of_mass()

    def calc_center_of_mass(self):
        if np.sum(self.img_slice[:, :, 3]) == 0:
            self.center_of_mass = np.array([0, 0])
        else:
            self.center_of_mass = scipy.ndimage.center_of_mass(self.img_slice[:, :, 3])