
import argparse
import logging
import os
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from matplotlib import pyplot as plt
from syclops_postprocessor.postprocessor_interface import PostprocessorInterface
try:
    from layer_collection import LayerCollection
except ModuleNotFoundError:
    from syclops_auge.layer_collection import LayerCollection
try:
    from aug_image.aug_image import AugImage
except ModuleNotFoundError:
    from syclops_auge.aug_image.aug_image import AugImage



class AugE(PostprocessorInterface):
    meta_description = {
        "type": "AUGMENTATION",
        "description": "Augmentations of the output."
    }

    def __init__(self, output_blender: str):
        """
        Manage augmentations.
        """
        super().__init__(output_blender)
        self.lc = LayerCollection()
        self.dir_in = Path(output_blender['parent_dir'])
        self.dir_temp = "temp"
        self.dir_out = os.path.join(self.dir_in, "augmented")
        self.dir_out_rgb = os.path.join(self.dir_out, "camera_main_camera", "rect")
        self.dir_out_instances = os.path.join(self.dir_out, "main_camera_annotations", "instance_segmentation")
        self.dir_out_semantics = os.path.join(self.dir_out, "main_camera_annotations", "semantic_segmentation")
        self.dir_in_rgb = os.path.join(self.dir_in, "main_camera", 'rect')
        os.makedirs(self.dir_out_rgb, exist_ok=True)
        os.makedirs(self.dir_out_instances, exist_ok=True)
        os.makedirs(self.dir_out_semantics, exist_ok=True)
        os.makedirs(self.dir_temp, exist_ok=True)

    def process_step(self, step_num: int, step_dict: dict) -> dict:
        self.add_data(step_num, step_dict)

    def process_all_steps(self) -> dict:
        self.finalize()

    def _output_folder_path(self) -> str:
        return self.dir_out

    def add_data(self, step_num, step_dict: dict):
        aug_img = AugImage.from_path(self.dir_in, step_num, step_dict, bg_component=1)
        self.lc.add_aug_img(aug_img, self.dir_temp)
        

    def finalize(self, p_use=1):
        """
        @param p_use:
        @return:
        """
        # generate new data
        for fname in os.listdir(self.dir_temp):
            with open(os.path.join(self.dir_temp, fname), 'rb') as f:
                aug_img = pickle.load(f)

            indices = list(range(1, len(aug_img.layers_draw)))
            indices = np.random.choice(indices, size=max(1, round(len(indices) * p_use)), replace=False)
            for idx in indices:
                layer = aug_img.layers_draw[idx]
                substitute = self.lc.get_substitute(layer)

                diff = np.array([a - b for a, b in zip(layer.center_of_mass, substitute.center_of_mass)])
                substitute.pos = layer.pos_orig + diff

                aug_img.layers_draw[idx] = substitute

            bgr, instances, semantics = aug_img.construct_img()

            # output data
            cv2.imwrite(os.path.join(self.dir_out_rgb, f'{aug_img.name_img}.png'), bgr)
            np.save(os.path.join(self.dir_out_instances, f'{aug_img.name_img}.npy'), instances)
            np.save(os.path.join(self.dir_out_semantics, f'{aug_img.name_img}.npy'), semantics)

        self.reset()

    def reset(self):
        # empty temp folder
        for filename in os.listdir(self.dir_temp):
            file_path = os.path.join(self.dir_temp, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pid",
        help="pid",
        type=int,
    )
    parser.add_argument(
        "--output",
        help="Path to blender output",
    )
    args = parser.parse_args()
    auge = AugE(args.output, args.pid)
    auge.run()
