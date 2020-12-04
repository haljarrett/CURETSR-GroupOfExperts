import random
import os
import torch
import torch.utils.data


import glob

from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def standardization(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    for t in tensor:
        t.sub_(t.mean()).div_(t.std())
        
    return tensor


def l2normalize(tensor):
    if not _is_tensor_image(tensor):
        raise TypeError('Tensor is not a torch image')

    tensor = tensor.mul(255)
    norm_tensor = tensor/torch.norm(tensor)
    return norm_tensor

# setup dataset class

class CURETSRDataset (torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, sequenceType, challengeType, challengeLevel, transform=None, target_transform=None,
                loader = pil_loader):
        self.dataset_dir = dataset_dir
        
        globpath = os.path.join(dataset_dir, f'*{split}/*/{sequenceType}_*_{challengeType}_{challengeLevel}_*.bmp')
#         print(globpath)
        self.paths = glob.glob(globpath)
        self.labels = [int(os.path.basename(path)[3:5]) - 1 for path in self.paths]
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.paths[index], self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.paths)


# deep learning vm has massive host memory - preloading to host memory may improve train time
class CURETSRDatasetCached (torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, sequenceType, challengeType, challengeLevel, fromPaths=None, transform=None, target_transform=None,
                loader = pil_loader):
        if fromPaths is None:
            self.dataset_dir = dataset_dir
        
            globpath = os.path.join(dataset_dir, f'*{split}/*/{sequenceType}_*_{challengeType}_{challengeLevel}_*.bmp')
            self.paths = glob.glob(globpath)
        else:
            self.paths = fromPaths
        self.labels = [int(os.path.basename(path)[3:5]) - 1 for path in self.paths]
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        self.images = [self.transform(self.loader(path)) for path in self.paths]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        image, target = self.images[index], self.labels[index]
        
        return image, target

    def __len__(self):
        return len(self.paths)

# use noise labels instead of class labels  
class CURETSRDataset_Noise_Cached (torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, sequenceType, challengeType, challengeLevel, fromPaths=None, transform=None, target_transform=None,
                loader = pil_loader):
        if fromPaths is None:
            self.dataset_dir = dataset_dir
        
            globpath = os.path.join(dataset_dir, f'*{split}/*/{sequenceType}_*_{challengeType}_{challengeLevel}_*.bmp')
            self.paths = glob.glob(globpath)
        else:
            self.paths = fromPaths
        
        # probably a better way to do this
        label_map = {}
        label_map['00'] = 0
        label_map['06'] = 1
        label_map['03'] = 2
        label_map['02'] = 3

        self.labels = [label_map[os.path.basename(path)[6:8]] for path in self.paths]
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        self.images = [self.transform(self.loader(path)) for path in self.paths]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        image, target = self.images[index], self.labels[index]
        
        return image, target

    def __len__(self):
        return len(self.paths)
    
    
class CURETSRDataset_Noise_AllLabels_Cached (torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, sequenceType, challengeType, challengeLevel, fromPaths=None, transform=None, target_transform=None,
                loader = pil_loader):
        if fromPaths is None:
            self.dataset_dir = dataset_dir
        
            globpath = os.path.join(dataset_dir, f'*{split}/*/{sequenceType}_*_{challengeType}_{challengeLevel}_*.bmp')
            self.paths = glob.glob(globpath)
        else:
            self.paths = fromPaths
        
        self.labels = [int(os.path.basename(path)[6:8]) for path in self.paths]
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
        self.images = [self.transform(self.loader(path)) for path in self.paths]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        image, target = self.images[index], self.labels[index]
        
        return image, target

    def __len__(self):
        return len(self.paths)