import os
import cv2

dir_rgb = r'C:\Users\HEW\Projekte\plant_disease_datasets\Maize whole plant image dataset\raw_images'
dir_masks = r'C:\Users\HEW\Projekte\plant_disease_datasets\Maize whole plant image dataset\segmented_images'
# dir_meta = r'C:\Users\HEW\Projekte\plant_disease_datasets\Maize whole plant image dataset\segmentationdata.csv'

for fname in os.listdir(dir_rgb):
    name, ending = fname.split('.')
    bgr = cv2.imread(os.path.join(dir_rgb, fname))
    mask = cv2.imread(os.path.join(dir_masks, 'bin_' + name + '.png'))
    bgr[mask==0] *= 0
    cv2.imwrite(f'out/maize_preprocessed/{name}_preprocessed.{ending}', bgr)


# Notes:
# Masken sind nicht perfekt. manchmal gehen Details verloren.
# Zum Teil sind die blauen plastik Bänder mit drin.