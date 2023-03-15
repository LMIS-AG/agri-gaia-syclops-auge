import os
import pickle
import shutil

import cv2
import numpy as np
from syclops_postprocessor.postprocessor_interface import PostprocessorInterface

try:
    from layer_collection import LayerCollection
except ModuleNotFoundError:
    from syclops_auge.layer_collection import LayerCollection
try:
    from aug_image.aug_image import AugImage
    from util import is_installed, run
except ModuleNotFoundError:
    from syclops_auge.aug_image.aug_image import AugImage
    from syclops_auge.util import is_installed, run


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
        self.sd_downscale = output_blender['sd_downscale']
        self.sd_inpaint = output_blender['sd_inpaint']
        if self.sd_inpaint:
            if not is_installed('torch') or not is_installed("torchvision"):
                torch_command = os.environ.get('TORCH_COMMAND',
                                               "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117")

                run(torch_command)
            if not is_installed('diffusers') or not is_installed("transformers"):
                command = "pip install diffusers[torch] transformers"
                run(command)

        self.dir_temp = "temp"
        os.makedirs(self.dir_temp, exist_ok=True)

    def _prepare(self):
        '''overwrite to setup paths and create Layer Collection'''
        self.lc = LayerCollection()
        paths = self.get_base_paths_by_type()
        for meta_type, path_in in paths.items():
            # path_in = os.path.dirname(path)
            path_rel = os.path.relpath(path_in, self.config['parent_dir'])
            path_out = os.path.join(self.output_folder, path_rel)
            os.makedirs(path_out)

            if meta_type == 'RGB':
                self.path_color_out = path_out
            elif meta_type == 'INSTANCE_SEGMENTATION':
                self.path_instance_out = path_out
            elif meta_type == 'SEMANTIC_SEGMENTATION':
                self.path_semantic_out = path_out
            elif meta_type == 'DEPTH':
                self.path_depth_out = path_out

    def process_step(self, step_num: int, step_dict: dict) -> dict:
        paths = self.get_full_paths_from_step_dict(step_dict)
        path_bgr = paths['RGB'][0]
        path_depth = paths['DEPTH'][0]
        path_instance = paths['INSTANCE_SEGMENTATION'][0]
        path_semantic = paths['SEMANTIC_SEGMENTATION'][0]

        aug_img = AugImage.from_path(path_bgr, path_depth, path_instance, path_semantic, name_img=str(step_num),
                                     bg_classes=self.config['bg_classes'], sd_inpaint=self.sd_inpaint, sd_downscale=self.sd_downscale)

        path_temp = os.path.join(self.dir_temp, f"{step_num}.pickle")
        with open(path_temp, "wb") as f:
            pickle.dump(aug_img, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.lc.add_aug_img(aug_img, self.config)

        output_step_dict = {step_num: [{"type": "TEMP",
                                        "path": path_temp}]}
        return output_step_dict

    def _output_folder_path(self) -> str:
        return os.path.join(self.config['parent_dir'], "augmented")

    def process_all_steps(self) -> dict:
        output_step_dict = {}
        for step_num, path_color, path_instance, path_semantic in self.generate_new_data():
            output_step_dict[step_num] = [{"type": "RGB",
                                           "path": path_color},
                                          {"type": "INSTANCE_SEGMENTATION",
                                           "path": path_instance},
                                          {"type": "SEMANTIC_SEGMENTATION",
                                           "path": path_semantic}]

        self.reset()
        return output_step_dict

    def generate_new_data(self, p_use=1):
        '''
        generate new data and yield paths
        @return: paths to ne datapoints
        '''
        # generate new data
        for fname in os.listdir(self.dir_temp):
            step_num = int(fname.rstrip('.pickle'))
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
            path_color = os.path.join(self.path_color_out, f'{aug_img.name_img}.png')
            path_instance = os.path.join(self.path_instance_out, f'{aug_img.name_img}.npy')
            path_semantic = os.path.join(self.path_semantic_out, f'{aug_img.name_img}.npy')
            cv2.imwrite(path_color, bgr)
            np.save(path_instance, instances)
            np.save(path_semantic, semantics)
            yield step_num, path_color, path_instance, path_semantic

    def reset(self):
        """
        clear the temp folder
        """
        for filename in os.listdir(self.dir_temp):
            file_path = os.path.join(self.dir_temp, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
