import argparse
import logging
import os
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
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
        self.bg_class = output_blender['bg_class']
        self.use_sd = output_blender['use_sd']
        self.dir_temp = "temp"
        os.makedirs(self.dir_temp, exist_ok=True)

    @classmethod
    def parse_paths(cls, input: dict, key='metadata',
                    expected_types=['RGB', 'INSTANCE_SEGMENTATION', 'SEMANTIC_SEGMENTATION', 'DEPTH']):
        parsed = {}
        expected_types_copy = [x for x in expected_types]
        for meta in input.values():
            meta_type = meta[key]['type']
            if meta_type not in expected_types_copy:
                continue
            expected_types_copy.remove(meta_type)
            if isinstance(meta, list):
                parsed[meta_type] = [x['path'] for x in meta]
            else:
                parsed[meta_type] = meta['path']
        assert len(expected_types_copy) == 0, "could not gather all neccessary output paths"
        return parsed

    def _prepare(self):
        '''overwrite to setup paths'''
        parsed = self.parse_paths(self.input_metadata)
        for meta_type, path in parsed.items():
            path_in = os.path.dirname(path)
            path_rel = os.path.relpath(path_in, self.config['parent_dir'])
            path_out = os.path.join(self.output_folder, path_rel)
            os.makedirs(path_out)

            if meta_type == 'RGB':
                self.path_color_in, self.path_color_out = path_in, path_out
            elif meta_type == 'INSTANCE_SEGMENTATION':
                self.path_instance_in, self.path_instance_out = path_in, path_out
            elif meta_type == 'SEMANTIC_SEGMENTATION':
                self.path_semantic_in, self.path_semantic_out = path_in, path_out
            elif meta_type == 'DEPTH':
                self.path_depth_in, self.path_depth_out = path_in, path_out

    def process_all_steps(self) -> dict:
        self.finalize()

    def process_step(self, step_num: int, step_dict: dict):
        parsed = self.parse_paths(step_dict, key=0)
        path_bgr = os.path.join(self.path_color_in, parsed['RGB'][0])
        path_depth = os.path.join(self.path_depth_in, parsed['DEPTH'][0])
        path_instance = os.path.join(self.path_instance_in, parsed['INSTANCE_SEGMENTATION'][0])
        path_semantic = os.path.join(self.path_semantic_in, parsed['SEMANTIC_SEGMENTATION'][0])
        aug_img = AugImage.from_path(step_num, self.bg_class, path_bgr, path_depth, path_instance, path_semantic)
        self.lc.add_aug_img(aug_img, self.dir_temp)

    def _output_folder_path(self) -> str:
        return os.path.join(self.config['parent_dir'], "augmented")

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
            cv2.imwrite(os.path.join(self.path_color_out, f'{aug_img.name_img}.png'), bgr)
            np.save(os.path.join(self.path_instance_out, f'{aug_img.name_img}.npy'), instances)
            np.save(os.path.join(self.path_semantic_out, f'{aug_img.name_img}.npy'), semantics)

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

