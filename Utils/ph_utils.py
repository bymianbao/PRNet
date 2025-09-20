import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def pil_to_tensor(image):
    return torch.tensor(np.array(image), dtype=torch.float32)

def load_images(image_paths, tfs=None):
    images = []
    for path in sorted(image_paths):
        image = Image.open(path)
        if tfs:
            image = tfs(image)
        image_tensor = pil_to_tensor(image).unsqueeze(0)
        images.append(image_tensor)
    volume_tensor = torch.cat(images, dim=0)
    volume_normalized = (volume_tensor - volume_tensor.min()) / (volume_tensor.max() - volume_tensor.min())
    return volume_normalized

class PhDatasetBase(Dataset):
    def __init__(self, csv_path, tfs=None, num_frames=16, path_prefix=''):
        self.data = pd.read_csv(csv_path)
        self.bag_list = self.data['image']
        self.labels = self.data['label']
        self.transform = tfs
        self.num_frames = num_frames
        self.path_prefix = path_prefix

    def __getitem__(self, index):
        folder_path = self.path_prefix + self.bag_list[index]
        image_files = glob.glob(folder_path + '/*.jpg')
        volume = load_images(image_files, tfs=self.transform)
        indices = torch.linspace(0, volume.size(0) - 1, steps=self.num_frames).long()
        volume = volume[indices].unsqueeze(0)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return volume, label

    def __len__(self):
        return len(self.data)

class PhDataset(PhDatasetBase):
    def __init__(self, csv_path, tfs=None):
        super().__init__(csv_path, tfs, num_frames=16)

class PhDataset_test(PhDatasetBase):
    def __init__(self, csv_path, tfs=None):
        super().__init__(csv_path, tfs, num_frames=16)

