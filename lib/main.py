import argparse
import os
import pickle
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
from matplotlib import pyplot as plt

from aug_image.aug_image import AugImage
from layer_collection import LayerCollection
from syclops_postprocessor.postprocessor_interface import PostprocessorInterface


class AugE(PostprocessorInterface):
    def __init__(self, output_blender: str):
        """
        Manage augmentations.
        """
        self.lc = LayerCollection()
        # self.pipe_in = PipeConnection(pipe_in, True, True)
        self.dir_in = Path(output_blender)
        self.dir_temp = "temp"
        self.dir_out_rgb = os.path.join(self.dir_in, "augmented", "camera_main_camera", "rect")
        self.dir_out_instances = os.path.join(self.dir_in, "augmented", "camera_main_camera_annotations", "instance_segmentation")
        self.dir_out_semantics = os.path.join(self.dir_in, "augmented", "camera_main_camera_annotations", "semantic_segmentation")
        self.dir_in_rgb = os.path.join(self.dir_in, "camera_main_camera", 'rect')
        os.makedirs(self.dir_out_rgb, exist_ok=True)
        os.makedirs(self.dir_out_instances, exist_ok=True)
        os.makedirs(self.dir_out_semantics, exist_ok=True)

    def process_step(self, step_num: int, step_dict: dict) -> dict:
        fname =
        try:
            self.add_data(fname.split('.')[0])
            processed.add(fname)
        except FileNotFoundError:
            time.sleep(self.poll_interval)

    def process_all_steps(self) -> dict:
        pass

    def _output_folder_path(self) -> str:
        pass

    def run(self):
        processed = set()
        left = set(os.listdir(self.dir_in_rgb))
        process_running = psutil.pid_exists(self.pid)
        while process_running or len(left) > 0:
            for fname in left:
                try:
                    self.add_data(fname.split('.')[0])
                    processed.add(fname)
                except FileNotFoundError:
                    time.sleep(self.poll_interval)

            left = set(os.listdir(self.dir_in_rgb)) - processed
            if len(left) == 0:
                time.sleep(self.poll_interval)
                process_running = psutil.pid_exists(self.pid)
                left = set(os.listdir(self.dir_in_rgb)) - processed

        self.finalize()
        self.reset()

    def add_data(self, name: str):
        aug_img = AugImage.from_path(self.dir_in, name, bg_component=99)
        self.lc.add_aug_img(aug_img)

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

                # aug_img.layers_draw.pop(idx)
                aug_img.layers_draw[idx] = substitute

            bgr, instances, semantics = aug_img.construct_img()
            plt.imshow(cv2.cvtColor(np.array(bgr, dtype=np.uint8), cv2.COLOR_BGRA2RGB))
            plt.show()
            cv2.imwrite(os.path.join(self.dir_out_rgb, f'{aug_img.name_img}.png'), bgr)
            np.save(os.path.join(self.dir_out_instances, f'{aug_img.name_img}.npy'), instances)
            np.save(os.path.join(self.dir_out_semantics, f'{aug_img.name_img}.npy'), semantics)


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
