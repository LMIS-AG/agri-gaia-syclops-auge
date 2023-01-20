# Created by HEW at 14.05.2021
import copy
import cv2
import numpy as np
from .utils import get_crop_slices, compose
from .layer import Layer

class AugImageBase:

    def __init__(self, layers: list[Layer], cut_off_value: int, name_img: str, img_shape: np.array):
        """

        :param layers: list of Layer elements
        :param cut_off_value:
        :param name_img: name of the image created from
        """
        layers.sort(key=lambda l: l.depth, reverse=True)  # sort by depth (rendering order)
        self.layers_draw: list[Layer] = layers
        self.layers_orig: list[Layer] = copy.deepcopy(layers)

        self.nr_layers: int = len(self.layers_orig)
        self.plot_data: tuple = (None, None, None)
        self.cut_off_value: int = cut_off_value
        self.name_img = name_img
        self.img_shape = img_shape
        self.add_layer_completeness()

    @classmethod
    def create_layers(cls, img: np.array, components_map: np.array, instances_map: np.array, depths_map: np.array,
                      bg_component: int, min_layer_area: float) -> list[Layer]:
        """
        create layers
        @param img: 
        @param components_map: 
        @param instances_map: 
        @param depths_map: 
        @param bg_component: 
        @param min_layer_area: 
        @return: 
        """

        def create_entry(mask: np.array) -> Layer:
            """
            create Layer from mask
            @param mask:
            """
            # crop to mask
            slice_x, slice_y = get_crop_slices(mask)
            img_slice, mask_slice = np.copy(img[slice_x, slice_y]), mask[slice_x, slice_y]

            # set alpha val of pixels not in mask to zero
            img_slice[np.logical_not(mask_slice), 3] *= 0
            depth = np.mean(depths_map[slice_x, slice_y][mask_slice])  # depth of that element todo: max mean median?

            # create entry
            layer = Layer(np.array((slice_x.start, slice_y.start), dtype=float), img_slice, component, depth)
            if layer.component == 0.0:  # todo patchwork fix weil Horizont (coponent 0) in den Daten Tiefe 0 hat
                layer.depth = np.inf
            return layer

        # calculate absolute min layer size and treat smaller layers as background
        min_layer_size = img.shape[0] * img.shape[1] * min_layer_area
        unique, counts = np.unique(instances_map, return_counts=True)
        remove = unique[counts < min_layer_size]
        mask_remove = np.any([instances_map == r for r in remove], axis=0)
        components_map[mask_remove] = bg_component

        # create background via inpainting
        img_inpaint = cls.create_inpaint_background(img, components_map, bg_component, do_visualize=False)
        layers = [Layer(np.array((0, 0), dtype=float), img_inpaint, bg_component,
                        np.inf)]  # set current background (priority 0)

        # get the background instance id from the component id
        bg_instances = set(instances_map[components_map == bg_component])
        for inst in set(instances_map.flat) - bg_instances:
            mask_inst = instances_map == inst
            component = components_map[mask_inst][0]
            layer = create_entry(mask_inst)
            layers.append(layer)
        return layers

    @staticmethod
    def create_inpaint_background(img: np.array, components_map: np.array, bg_class: int, ksize: int = 5,
                                  inpaint_radius: int = 5, do_visualize: bool = False) -> np.array:
        '''
        use opencvs inpaint to create a pure background image (without foreground elements)
        :param img:
        :param components_map:
        :param bg_class: the class that marks the background
        :param ksize: kernel size for dilation of foreground mask
        :param inpaint_radius: parameter for inpaint algorithm
        :param do_visualize: visualize result yay or ney
        :return:
        '''
        fg_mask = np.array(components_map != bg_class, dtype=np.uint8) * 255
        kernel = np.ones((ksize, ksize), np.uint8)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)
        img_inpaint = cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR),
                                  fg_mask,
                                  inpaint_radius,
                                  cv2.INPAINT_NS)
        if do_visualize:
            cv2.imshow('inpaint', img_inpaint)
            orig = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            orig[fg_mask > 0] *= 0
            cv2.imshow('original', orig)
            cv2.imshow('foreground_mask', fg_mask)
            cv2.waitKey()
        img_inpaint = cv2.cvtColor(img_inpaint, cv2.COLOR_BGR2BGRA)
        return img_inpaint

    def construct_img(self):
        '''
        construct the final image from the layers
        :return:
        '''
        full_img = np.zeros(tuple(self.img_shape))  # same shape as background
        instances = np.zeros((self.img_shape[0], self.img_shape[1]), dtype=np.uint)
        semantics = np.zeros((self.img_shape[0], self.img_shape[1]), dtype=np.uint)
        layer_components = []
        for label, layer in enumerate(self.layers_draw):
            pos, img_slice, component = layer.pos, layer.img_slice, layer.component

            layer_components.append(component)
            pos = (round(pos[0]), round(pos[1]))  # discretize
            s_x, s_y, img_slice = compose(full_img, img_slice, pos)
            instances[s_x, s_y][img_slice[:, :, 3] > 0] = label
            semantics[s_x, s_y][img_slice[:, :, 3] > 0] = component
        return full_img, instances, semantics

    def add_layer_completeness(self):
        """
        check whether the layers object can be considered complete
        @return:
        """
        img, instances, _ = self.construct_img()
        masks_dilated = []
        for label in range(len(self.layers_draw) - 1, -1, -1):
            l = self.layers_draw[label]
            if l.component == 0 or l.depth == np.inf:  # todo 0 ist der Fehlerfall in den Daten, inf depth heisst Hintergrund
                l.is_complete = False
                continue

            mask = instances == label
            is_incomplete = np.any(mask[0, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]) or np.any(mask[-1, :])

            # check whether object is not occluded by a a closer object
            for mask_dilated in masks_dilated:
                if is_incomplete:
                    break
                is_incomplete = np.any(np.logical_and(mask_dilated, mask))

            mask_dilated = cv2.dilate(np.array(mask, dtype=np.uint8), np.ones((3, 3), np.uint8)) > 0
            masks_dilated.append(mask_dilated)

            l.is_complete = not is_incomplete
            self.layers_orig[label].is_complete = not is_incomplete
