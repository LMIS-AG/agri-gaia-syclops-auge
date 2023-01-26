import os
import pickle
import shutil
from os import listdir
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm

from aug_image.aug_image import AugImage


class DataLoader:
    def __init__(self, path: str, step_exclude):
        """

        :param path: path to the dataset
        :param p: percentage of data to be used (defaults to 1: all data)
        """

        path_main_cam = join(path, "main_camera")
        path_main_cam_ann = join(path, "main_camera_annotations")
        self.path_pickle = os.path.join(path, 'AugImages')
        self.labels = self.load_labels(join(path, 'labels.txt'))
        self.step_exclude = step_exclude
        if os.path.isdir(self.path_pickle):
            print("Try to read from pickled AugImages")
            nr_data = len(listdir(self.path_pickle))

            self.len = round(nr_data - (nr_data // step_exclude)) if step_exclude else nr_data
            self.from_pickled = True
        else:
            print("Try to read rendered Data")
            self.bgr = join(path_main_cam, 'rect')
            self.depth = join(path_main_cam_ann, 'depth')
            self.instance = join(path_main_cam_ann, 'instance_segmentation')
            self.component = join(path_main_cam_ann, 'semantic_segmentation')

            nr_data = len(listdir(self.bgr))
            self.len = round(nr_data - (nr_data // step_exclude)) if step_exclude else nr_data

            self.from_pickled = False


    def generate_data(self):
        names_png = sorted(listdir(self.bgr))
        for i, name_png in enumerate(names_png):
            name_img = name_png.rstrip('.png')
            name_npy = name_img + '.npz'
            bgr = cv2.imread(join(self.bgr, name_png))
            depth = np.load(join(self.depth, name_npy))['array']
            instance = np.load(join(self.instance, name_npy))['array']
            component = np.load(join(self.component, name_npy))['array']
            yield bgr, depth, instance, component, name_img

    def generate_aug_images(self):
        if self.from_pickled:
            for i, fname in tqdm(enumerate(sorted(os.listdir(self.path_pickle))), total=len(self)):
                if self.step_exclude and (i + 1) % self.step_exclude == 0:
                    continue
                with open(os.path.join(self.path_pickle, fname), 'rb') as f:
                    aug_img = pickle.load(f)
                yield aug_img

        else:
            for i, data in tqdm(enumerate(self.generate_data()), total=len(self)):
                if self.step_exclude and (i + 1) % self.step_exclude == 0:
                    continue
                bgr, depth, instance, component, name_img = data
                # plt.figure(), plt.imshow(bgr), plt.figure(), plt.imshow(component), plt.show()
                # img_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                img_bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                aug_img = AugImage.from_data(img_bgra,
                                             instance,
                                             component,
                                             depth,
                                             name_img,
                                             bg_component=99)
                yield aug_img

    @staticmethod
    def pickle_aug_img(path, aug_img):
        os.makedirs(path, exist_ok=True)
        files = os.listdir(path)
        name = aug_img.name_img
        nr = 1
        while (name_full := name + '.pickle') in files:
            name = f"{aug_img.name_img}_{str(nr)}"
            nr += 1
        with open(os.path.join(path, name_full), 'wb') as f:
            pickle.dump(aug_img, f)

    def pickle_dataset_as_aug_images(self):
        for aug_img in self.generate_aug_images():
            self.pickle_aug_img(self.path_pickle, aug_img)

    def __iter__(self):
        return self.generate_aug_images()

    def __len__(self):
        return self.len

    @staticmethod
    def load_labels(path):
        labels = {}
        class_in_to_class_out = {}

        with open(path, 'r') as f:
            for line in f.readlines():
                label_class_in, label_text = line.split(',')
                label_class_in, label_text = int(label_class_in), label_text.strip()
                try:
                    label_class_out = class_in_to_class_out[label_class_in]
                except KeyError:
                    label_class_out = max(class_in_to_class_out.values()) if class_in_to_class_out else 0
                    class_in_to_class_out[label_class_in] = label_class_out

                labels[label_class_in] = (label_text, label_class_out)
        return labels

