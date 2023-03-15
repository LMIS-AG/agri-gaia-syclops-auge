import pickle
import sqlite3 as sl
from os.path import join

import numpy as np

try:
    from aug_image.aug_image import AugImage
    from aug_image.layer import Layer
except ModuleNotFoundError:
    from syclops_auge.aug_image.aug_image import AugImage
    from syclops_auge.aug_image.layer import Layer
from scipy.ndimage import center_of_mass


class LayerCollection:
    '''
    Collection of complete instance layers.
    Layers are already augmented when added to the collection.
    Store layers together with their label class and their pose.
    '''

    def __init__(self, db_path='', db_name='layers.db'):
        self.con = sl.connect(join(db_path, db_name))
        table_name = "LAYERS"
        try:
            with self.con:
                self.con.executescript(f"""
                    DROP TABLE IF EXISTS LAYERS;
                    CREATE TABLE {table_name} (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        classlabel INTEGER,
                        pixel_size INTEGER,
                        layer BLOB,
                        com_x REAL,
                        com_y REAL,
                        pos_x REAL,
                        pos_y REAL);""")
        except sl.OperationalError:
            print(f"table {table_name} already exists, no new table created")

    def add(self, classlabel: int, layer: np.array, pos: np.array):
        com = center_of_mass(layer[:, :, 3])
        with self.con:
            res = self.con.execute("INSERT INTO LAYERS(classlabel, pixel_size, layer, com_x, com_y, pos_x, pos_y) "
                                   "VALUES(?,?,?,?,?,?,?)",
                                   (int(classlabel), layer.size, layer.dumps(), com[0], com[1], pos[0], pos[1]))

    def add_aug_img(self, aug_img: AugImage, config: dict, n=1):

        # if config['stable_diffusion']['sd_rust_disease']:
        #     aug_img.sd_augment(config['target_classes'])

        targets = [(i, l) for i, l in enumerate(aug_img.layers_draw) if l.is_complete]
        for _, l in targets:
            self.add(l.component, l.img_slice, (l.pos[0], aug_img.img_shape[1] - l.pos[1]))
        target_indices = [i for i, _ in targets]
        for do_flip in [False, True]:
            for _ in range(n):
                aug_img.perform_targets_random(np.array(target_indices), p_flip=do_flip * 1)
                for i in target_indices:
                    l_new = aug_img.layers_draw[i]
                    self.add(l_new.component, l_new.img_slice, (l_new.pos[0], aug_img.img_shape[1] - l_new.pos[1]))

    def get_substitute(self, layer: Layer) -> Layer:
        """
        find and return appropriate substitute for given layer in DB
        @param layer: layer to substituted
        @return: layer substitute
        """
        candidates = self.con.execute(f"SELECT * FROM LAYERS WHERE classlabel={int(layer.component)}").fetchall()
        # f" AND pixel_size > {layer.size / 2} AND pixel_size < {layer.size * 2}").fetchall()
        if candidates:
            probs = [1 / (np.sqrt(np.sum(np.square(layer.pos - np.array([c[6], c[7]])))) + 1) for c in candidates]
            chosen_idx = np.random.choice(range(len(candidates)), p=probs / np.sum(probs))
            chosen = candidates[chosen_idx]
            l = Layer((chosen[6], chosen[7]), pickle.loads(chosen[3]), chosen[1], layer.depth)
            return l
        else:
            return layer

