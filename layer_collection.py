import pickle
import sqlite3 as sl
from os.path import join

import numpy as np
from aug_image.aug_image import AugImage
from aug_image.layer import Layer
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
--                     DROP TABLE IF EXISTS LAYERS;
                    CREATE TABLE {table_name} (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        classlabel INTEGER,
                        layer BLOB,
                        com_x REAL,
                        com_y REAL,
                        pos_x REAL,
                        pos_y REAL);""")
        except sl.OperationalError:
            print(f"table {table_name} already exists, no new table created")

    def add(self, classlabel:int, layer:np.array, pos:np.array):
        com = center_of_mass(layer[:, :, 3])
        with self.con:
            self.con.execute("INSERT INTO LAYERS(classlabel, layer, com_x, com_y, pos_x, pos_y) VALUES(?,?,?,?,?,?)",
                             (classlabel, layer.dumps(), com[0], com[1], pos[0], pos[1]))

    def add_aug_img(self, aug_img: AugImage):
        for l in aug_img.layers_draw:
            self.add(l.component, l.img_slice, l.pos)

    def get(self, classlabel):
        # con = sl.connect(":memory:")
        self.con.row_factory = self.dict_factory
        # cur = self.con.cursor()
        # cur.execute("select 1 as a")
        # cur.fetchone()["a"]
        res = self.con.execute(f"SELECT * FROM LAYERS WHERE classlabel='{classlabel}'").fetchall()
        for d in res:
            d['layer'] = pickle.loads(d['layer'])
        return res

    @staticmethod
    def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

if __name__ == '__main__':
    r1d = np.random.randn(100, 200, 4)
    lc = LayerCollection()
    lc.add('test', r1d, np.array([-3, 4.05]))
    res = lc.get('test')
    print(res)
