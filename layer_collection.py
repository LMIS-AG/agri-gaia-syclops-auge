import pickle
import sqlite3 as sl
from os.path import join

import numpy as np

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
                self.con.execute(f"""
--                     DROP TABLE IF EXISTS LAYERS;
                    CREATE TABLE {table_name} (
                        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                        classlabel INTEGER,
                        layer BLOB);
                    """)
        except sl.OperationalError:
            print(f"table {table_name} already exists, no new table created")

    def add(self, classlabel, layer):
        with self.con:
            self.con.execute("INSERT INTO LAYERS(classlabel,layer) VALUES(?,?)", (classlabel, layer.dumps()))

    def get(self, classlabel):
        with self.con:
            res = self.con.execute(f"SELECT layer FROM LAYERS WHERE classlabel='{classlabel}'").fetchall()
        res_arr = [pickle.loads(r[0]) for r in res]
        return res_arr

if __name__ == '__main__':
    r1d = np.random.randn(10, 10)
    lc = LayerCollection()
    lc.add('test', r1d)
    res = lc.get('test')
    print(res)
