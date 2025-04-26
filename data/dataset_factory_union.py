""" Dataset Factory

Hacked together by / Copyright 2021, Ross Wightman
"""
import os
#import hub

from torchvision.datasets import CIFAR100, CIFAR10, MNIST, QMNIST, KMNIST, FashionMNIST, ImageNet, ImageFolder
try:
    from torchvision.datasets import Places365
    has_places365 = True
except ImportError:
    has_places365 = False
try:
    from torchvision.datasets import INaturalist
    has_inaturalist = True
except ImportError:
    has_inaturalist = False

from timm.data.dataset import IterableImageDataset, ImageDataset

import urllib
from PIL import Image
import os
from typing import Dict, List, Optional, Set, Tuple, Union

from timm.utils.misc import natural_key

from .class_map import load_class_map
from .img_extensions import get_img_extensions
from .reader import Reader
def find_images_and_targets(
        folder: str,
        types: Optional[Union[List, Tuple, Set]] = None,
        class_to_idx: Optional[Dict] = None,
        leaf_name_only: bool = True,
        sort: bool = True
):
    """ Walk folder recursively to discover images and map them to classes by folder names.

    Args:
        folder: root of folder to recrusively search
        types: types (file extensions) to search for in path
        class_to_idx: specify mapping for class (folder name) to class index if set
        leaf_name_only: use only leaf-name of folder walk for class names
        sort: re-sort found images by name (for consistent ordering)

    Returns:
        A list of image and target tuples, class_to_idx mapping
    """
    types = get_img_extensions(as_set=True) if not types else set(types)
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False, followlinks=True):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx

class ParserImageFolder(Parser):

    def __init__(
            self,
            root,
            class_map=''):
        super().__init__()

        self.root = root
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        self.samples, self.class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(self.samples) == 0:
            raise RuntimeError(
                f'Found 0 images in subfolders of {root}. Supported image extensions are {", ".join(IMG_EXTENSIONS)}')

    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

class CustomImageDataset(ImageDataset):

    def __init__(
            self,
            root,
            parser=None,
            class_map='',
            load_bytes=False,
            transform=None,
    ):
        super().__init__(root, parser=parser, class_map=class_map, load_bytes=load_bytes, **kwargs)
        if parser is None or isinstance(parser, str):
            parser = create_parser(parser or '', root=root, class_map=class_map)
        self.parser = parser
        self.load_bytes = load_bytes
        self.transform = transform
        self._consecutive_errors = 0

    def __getitem__(self, index):
        img, target = self.parser[index]
        try:
            img = img.read() if self.load_bytes else Image.open(img).convert('RGB')
        except Exception as e:
            _logger.warning(f'Skipped sample (index {index}, file {self.parser.filename(index)}). {str(e)}')
            self._consecutive_errors += 1
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % len(self.parser))
            else:
                raise e
        self._consecutive_errors = 0
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.tensor(-1, dtype=torch.long)
        return img, target

    def __len__(self):
        return len(self.parser)

    def filename(self, index, basename=False, absolute=False):
        return self.parser.filename(index, basename, absolute)

    def filenames(self, basename=False, absolute=False):
        return self.parser.filenames(basename, absolute)


class CustomImageDataset(ImageDataset):
    def __init__(self, root, parser=None, class_map=None, load_bytes=False, **kwargs):
        super().__init__(root, parser=parser, class_map=class_map, load_bytes=load_bytes, **kwargs)

        
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.parser is not None:
            sample = self.parser(sample)
        if self.load_bytes:
            sample = sample.tobytes()
        if self.class_map is not None:
            target = self.class_map[target]
        return sample, target

# my datasets
from .stanford_dogs import dogs
from .nabirds import NABirds
from .cub2011 import Cub2011
from .vtab import VTAB
from .vpt_dataset import CarsDataset, FlowersDataset


_TORCH_BASIC_DS = dict(
    cifar10=CIFAR10,
    cifar100=CIFAR100,
    mnist=MNIST,
    qmist=QMNIST,
    kmnist=KMNIST,
    fashion_mnist=FashionMNIST,
)
_TRAIN_SYNONYM = {'train', 'training'}
_EVAL_SYNONYM = {'val', 'valid', 'validation', 'eval', 'evaluation'}

_VTAB_DATASET = ['caltech101', 'clevr_count', 'dmlab', 'dsprites_ori', 'eurosat', 'flowers102', 'patch_camelyon', 'smallnorb_azi', 'svhn', 'cifar100', 'clevr_dist', 'dsprites_loc', 'dtd', 'kitti', 'pets', 'resisc45', 'smallnorb_ele', 'sun397', 'diabetic_retinopathy']




def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root


def create_dataset(
        name,
        root,
        split='validation',
        search_split=True,
        class_map=None,
        load_bytes=False,
        is_training='train',
        download=False,
        batch_size=None,
        repeats=0,
        **kwargs
):
    """ Dataset factory method

    In parenthesis after each arg are the type of dataset supported for each arg, one of:
      * folder - default, timm folder (or tar) based ImageDataset
      * torch - torchvision based datasets
      * TFDS - Tensorflow-datasets wrapper in IterabeDataset interface via IterableImageDataset
      * all - any of the above

    Args:
        name: dataset name, empty is okay for folder based datasets
        root: root folder of dataset (all)
        split: dataset split (all)
        search_split: search for split specific child fold from root so one can specify
            `imagenet/` instead of `/imagenet/val`, etc on cmd line / config. (folder, torch/folder)
        class_map: specify class -> index mapping via text file or dict (folder)
        load_bytes: load data, return images as undecoded bytes (folder)
        download: download dataset if not present and supported (TFDS, torch)
        is_training: create dataset in train mode, this is different from the split.
            For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS)
        batch_size: batch size hint for (TFDS)
        repeats: dataset repeats per iteration i.e. epoch (TFDS)
        **kwargs: other args to pass to dataset

    Returns:
        Dataset object
    """
    name = name.lower()
    if name.startswith('torch/'):
        name = name.split('/', 2)[-1]
        torch_kwargs = dict(root=root, download=download, **kwargs)
        if name in _TORCH_BASIC_DS:
            ds_class = _TORCH_BASIC_DS[name]
            use_train = split in _TRAIN_SYNONYM
            ds = ds_class(train=use_train, **torch_kwargs)
        elif name == 'inaturalist' or name == 'inat':
            assert has_inaturalist, 'Please update to PyTorch 1.10, torchvision 0.11+ for Inaturalist'
            target_type = 'full'
            split_split = split.split('/')
            if len(split_split) > 1:
                target_type = split_split[0].split('_')
                if len(target_type) == 1:
                    target_type = target_type[0]
                split = split_split[-1]
            if split in _TRAIN_SYNONYM:
                split = '2021_train'
            elif split in _EVAL_SYNONYM:
                split = '2021_valid'
            ds = INaturalist(version=split, target_type=target_type, **torch_kwargs)
        elif name == 'places365':
            assert has_places365, 'Please update to a newer PyTorch and torchvision for Places365 dataset.'
            if split in _TRAIN_SYNONYM:
                split = 'train-standard'
            elif split in _EVAL_SYNONYM:
                split = 'val'
            ds = Places365(split=split, **torch_kwargs)
        elif name == 'imagenet':
            if split in _EVAL_SYNONYM:
                split = 'val'
            ds = ImageNet(split=split, **torch_kwargs)
        elif name == 'image_folder' or name == 'folder':
            # in case torchvision ImageFolder is preferred over timm ImageDataset for some reason
            if search_split and os.path.isdir(root):
                # look for split specific sub-folder in root
                root = _search_split(root, split)
            ds = ImageFolder(root, **kwargs)
        else:
            assert False, f"Unknown torchvision dataset {name}"
    elif name.startswith('tfds/'):
        ds = IterableImageDataset(
            root, parser=name, split=split, is_training=is_training,
            download=download, batch_size=batch_size, repeats=repeats, **kwargs)
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future

        # define my datasets
        if name == 'stanford_dogs':
            ds = dogs(root=root, train=is_training, **kwargs)
        elif name == 'nabirds':
            ds = NABirds(root=root, train=is_training, **kwargs)
        elif name == 'cub2011':
            ds = Cub2011(root=root, train=is_training, **kwargs)
        elif name == 'stanford_cars' and 'ssf' not in is_training:
            print('constructing cars dataset')
            ds = CarsDataset(split=is_training)
            print('done constructing cars dataset')
        # elif name == 'oxford_flowers':# and 'ssf' not in is_training:
        #     print('constructing flowers dataset with validation split')
        #     ds = FlowersDataset(split=is_training)
        #     print('done constructing flowers dataset')
            
        elif name in _VTAB_DATASET:
            if name == 'clevr_dist':
                
                ds = VTAB(root=root, train=is_training, **kwargs)
            else:
                ds = VTAB(root=root, train=is_training, **kwargs)
        else:
            if os.path.isdir(os.path.join(root, split)):
                root = os.path.join(root, split)
            else:
                if search_split and os.path.isdir(root):
                    root = _search_split(root, split)
            # print(root)
            # exit()
            ds = ImageDataset(root, parser=name, class_map=class_map, load_bytes=load_bytes, **kwargs)
    return ds

