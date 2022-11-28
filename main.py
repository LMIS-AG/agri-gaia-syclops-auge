import cv2
import numpy as np
from matplotlib import pyplot as plt

from aug_image.aug_image import AugImage
from layer_collection import LayerCollection

class AugE:
    def __init__(self, job_description: dict):
        self.path = job_description['output_path']
        self.lc = LayerCollection()

    def add_data(self, name: str):
        aug_img = AugImage.from_path(self.path, name)
        img, labels = aug_img.construct_img()
        plt.imshow(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGRA2RGB))
        plt.show()
        self.lc.add_aug_img(aug_img)

    def finalize(self):
        breakpoint()
        pass




if __name__ == '__main__':
    jd = {'output_path': r'C:\tmp\output\2022_09_09_11_37_00'}
    auge = AugE(jd)
    auge.add_data('0000')
    auge.add_data('0001')
    auge.add_data('0002')
    auge.add_data('0003')
    auge.finalize()

