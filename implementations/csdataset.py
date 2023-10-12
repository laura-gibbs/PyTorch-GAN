import glob
import os
# from models import vae
# from torchvision.datasets import MNIST
# from torchvision import DataLoader
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision.transforms import ToTensor
from PIL import Image
import torch
import yaml

class CSDataset(Dataset):

    def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.paths = glob.glob(os.path.join(root_dir, '*.png'))
            print(root_dir)
            with open(os.path.join(root_dir, "_info.yml"), "r") as stream:
                metadata = yaml.safe_load(stream)
            self.tile_size = metadata['tile_size_degrees'] * 4
            self.overlap = metadata['overlap_degrees'] * 4
            self.fnames = metadata['fnames'] * 4
             

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_name = self.paths[idx]
        image = Image.open(img_name)
        # print(ToTensor()(image).min(), ToTensor()(image).max(), ToTensor()(image).mean(), ToTensor()(image).std())
        if self.transform is not None:
            image = self.transform(image)
        # print(image.min(), image.max(), image.mean(), image.std())
        return image, torch.tensor(0)
