import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import glob
import os
import json
from typing import List, Dict

standard_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

_IMAGENET_DATA_ROOT = '/fs/cml-datasets/ImageNet/ILSVRC2012/'

class Breeds(Dataset):
    """
    Breeds is a family of datasets, where each consists of sets supercategories of ImageNet classes.
    For example, Living17 consists of 17 supercategories from ImageNet, with four ImageNet classes making up each supercategory. 

    Options for dsetname are entity13, entity30, living17, nonliving26
    """

    def __init__(
        self, 
        dsetname: str = 'living17', 
        transform = standard_transform, 
        data_dir: str = _IMAGENET_DATA_ROOT, 
        split: str = 'val', 
    ):
        ''' data_dir should correspond to your ImageNet path, as Living17 is a subset of it. '''
        self.imagenet_dir = data_dir
        self.split = split
        self.dsetname = dsetname

        with open('./datasets/breeds_metafiles/inet_labels.txt') as f:
            inet_wnid_and_labels = f.readlines()
        self.inet_wnids = [wnid_and_label.split(',')[0] for wnid_and_label in inet_wnid_and_labels]

        self.transform = transform

        # breeds_info will have a dictionary w/ key for each dataset within breeds (e.g. 'living17' is a key)
        # breeds_info['living17'] will return another dictionary with living17 classnames as keys and 
        # ImageNet class idx as values (specifically, the classes in ImageNet that make up the Breeds class)
        with open('./datasets/breeds_metafiles/breeds_info.json', 'r') as f:
            breeds_info = json.load(f)
        self.breeds_classes_to_inet_cls_idx = breeds_info[dsetname]

        self.classes = list(self.breeds_classes_to_inet_cls_idx.keys())
        self.classname_to_label_dict = dict({c:label for label,c in enumerate(self.classes)})

        self.data_df = self.collect_instances()

        ### We will save the list of identifier strings (image_paths) now, at initialization.
        ### THIS SHOULD REMAIN STATIC. Same with self.classnames 
        self.static_input_path_list = self.data_df.index.tolist()

    def collect_instances(self):
        img_paths, classnames, labels = [], [], []

        # Recall, each Breeds class consists of the union of images from a set of ImageNet classes
        # inet_cls_idx_in_class is the set of ImageNet classes that compose the current Breeds class
        # 'classname' is the name of the current *Breeds* class
        for classname, inet_cls_idx_in_class in self.breeds_classes_to_inet_cls_idx.items():
            for inet_cls_ind in inet_cls_idx_in_class:

                curr_img_paths = glob.glob(os.path.join(self.imagenet_dir,self.split,self.inet_wnids[inet_cls_ind],'*'))
                img_paths.extend(curr_img_paths)
                N = len(curr_img_paths)

                classnames.extend([classname] * N)
                label = self.classname_to_label_dict[classname]
                labels.extend([label] * N)

        data_df = pd.DataFrame(list(zip(img_paths, classnames, labels)), 
                                    columns=['img_path', 'classname', 'label'])
        data_df = data_df.set_index('img_path')
        return data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, ind: int):
        if type(ind) is int:
            img_path = self.static_input_path_list[ind]
        else:
            img_path = ind

        row = self.data_df.loc[img_path]
        label, classname = [row[x] for x in ['label', 'classname']]

        img = Image.open(img_path)

        img_shape = np.array(img).shape
        if len(img_shape) != 3 or img_shape[2] != 3:
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

class Living17(Breeds):
    def __init__(self, split: str = 'val', transform = standard_transform):
        super().__init__(dsetname='living17', split=split, transform=transform)

class Nonliving26(Breeds):
    def __init__(self, split: str = 'val', transform = standard_transform):
        super().__init__(dsetname='nonliving26', split=split, transform=transform)

class Entity13(Breeds):
    def __init__(self, split: str = 'val', transform = standard_transform):
        super().__init__(dsetname='entity13', split=split, transform=transform)

class Entity30(Breeds):
    def __init__(self, split: str = 'val', transform = standard_transform):
        super().__init__(dsetname='entity30', split=split, transform=transform)