#!/usr/bin/env python3

"""JSON dataset: support CUB, NABrids, Flower, Dogs and Cars"""

import os
import torch
import torch.utils.data
import torchvision as tv
import numpy as np
from collections import Counter
from typing import List, Union
import json
from torchvision.datasets.folder import default_loader

# from ..transforms import get_transforms
# from ...utils import logging

def read_json(filename: str) -> Union[list, dict]:
    """read json files"""
    with open(filename, "rb") as fin:
        data = json.load(fin, encoding="utf-8")
    return data
# logger = logging.get_logger("visual_prompt")


class JSONDataset(torch.utils.data.Dataset):
    def __init__(self,split,name='none'):
        assert split in {
            "train",
            "val",
            "test",
        }, "Split '{}' not supported for {} dataset".format(
            split, 'cars')
        # logger.info("Constructing {} dataset {}...".format(
        #     'cars', split))

        self._split = split
        self.data_dir = '/data/datasets/FGVC/stanford_cars'
        if name == 'flowers':
            self.data_dir = '/data/datasets/FGVC/flowers_val'
        self.data_percentage = 1.0
        self._construct_imdb()
        # self.transform = get_transforms(split, cfg.DATA.CROPSIZE)

    def get_anno(self):
        anno_path = os.path.join(self.data_dir, "{}.json".format(self._split))
        if "train" in self._split:
            if self.data_percentage < 1.0:
                anno_path = os.path.join(
                    self.data_dir,
                    "{}_{}.json".format(self._split, self.data_percentage)
                )
        assert os.path.exists(anno_path), "{} dir not found".format(anno_path)

        return read_json(anno_path)

    def get_imagedir(self):
        raise NotImplementedError()

    def _construct_imdb(self):
        """Constructs the imdb."""

        img_dir = self.get_imagedir()
        assert os.path.exists(img_dir), "{} dir not found".format(img_dir)

        anno = self.get_anno()
        # Map class ids to contiguous ids
        self._class_ids = sorted(list(set(anno.values())))
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}

        # Construct the image db
        self._imdb = []
        for img_name, cls_id in anno.items():
            cont_id = self._class_id_cont_id[cls_id]
            # if self._split == "train" or "val" in self._split:
            #     # img_name = img_name.split("/")[-1]
            #     # img_name = 'cars_train/' + img_name
            # else: 
            #     img_name = img_name.split("/")[-1]
            #     img_name = 'cars_test/' + img_name
            im_path = os.path.join(img_dir, img_name)
                
            self._imdb.append({"im_path": im_path, "class": cont_id})
        print(f'{self._split} : {len(self._imdb)}')
        # logger.info("Number of images: {}".format(len(self._imdb)))
        # logger.info("Number of classes: {}".format(len(self._class_ids)))

    def get_info(self):
        num_imgs = len(self._imdb)
        return num_imgs, self.get_class_num()

    def get_class_num(self):
        return 196 #self.cfg.DATA.NUMBER_CLASSES
        # return len(self._class_ids)

    def get_class_weights(self, weight_type):
        """get a list of class weight, return a list float"""
        if "train" not in self._split:
            raise ValueError(
                "only getting training class distribution, " + \
                "got split {} instead".format(self._split)
            )

        cls_num = self.get_class_num()
        if weight_type == "none":
            return [1.0] * cls_num

        id2counts = Counter(self._class_ids)
        assert len(id2counts) == cls_num
        num_per_cls = np.array([id2counts[i] for i in self._class_ids])

        if weight_type == 'inv':
            mu = -1.0
        elif weight_type == 'inv_sqrt':
            mu = -0.5
        weight_list = num_per_cls ** mu
        weight_list = np.divide(
            weight_list, np.linalg.norm(weight_list, 1)) * cls_num
        return weight_list.tolist()

    def __getitem__(self, index):
        # Load the image
        
        im = tv.datasets.folder.default_loader(self._imdb[index]["im_path"])
        label = self._imdb[index]["class"]
        # im = self.transform(im)
        if self._split == "train":
            index = index
        else:
            index = f"{self._split}{index}"
        # sample = {
        #     "image": im,
        #     "label": label,
        #     # "id": index
        # }
        
        if self.transform is not None:
            img = self.transform(im)
        # if self.target_transform is not None:
        #     target = self.target_transform(label)
        # print(self.transform)
        # print(img)
        # exit()
        return img, label

    def __len__(self):
        return len(self._imdb)


class CUB200Dataset(JSONDataset):
    """CUB_200 dataset."""

    def __init__(self, cfg, split):
        super(CUB200Dataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")


class CarsDataset(JSONDataset):
    """stanford-cars dataset."""

    def __init__(self, split):
        super(CarsDataset, self).__init__(split)

    def get_imagedir(self):
        return self.data_dir

class FlowersDataset(JSONDataset):
    """flowers dataset."""

    def __init__(self, split):
        super(FlowersDataset, self).__init__(split,name='flowers')

    def get_imagedir(self):
        return self.data_dir

class DogsDataset(JSONDataset):
    """stanford-dogs dataset."""

    def __init__(self, cfg, split):
        super(DogsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "Images")




class NabirdsDataset(JSONDataset):
    """Nabirds dataset."""

    def __init__(self, cfg, split):
        super(NabirdsDataset, self).__init__(cfg, split)

    def get_imagedir(self):
        return os.path.join(self.data_dir, "images")