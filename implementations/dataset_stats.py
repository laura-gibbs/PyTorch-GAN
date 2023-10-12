import sys
import numpy as np
sys.path.append('./')
from csdataset import CSDataset

dataset = CSDataset(root_dir='C:/Users/oa18724/Documents/Master_PhD_folder/MDT-Calculations/saved_tiles/training/rescaled_tiles/')
images = [np.array(image)/255 for image, _ in dataset]
print(np.mean(images), np.std(images))


