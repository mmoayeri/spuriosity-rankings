import glob
import os
from datetime import datetime

from torch.utils import data
from torchvision.datasets.folder import pil_loader


### Original dataset webpage: https://susanqq.github.io/UTKFace/

class UTKFace(data.Dataset):

    gender_map = dict(male=0, female=1)
    race_map = dict(white=0, black=1, asian=2, indian=3, others=4)

    def __init__(self, root='/fs/cml-datasets/utkface/', transform=None, target='race', split='train'):
        self.root = root
        self.transform = transform
        self.split = split
        self.target = target
        self.samples = self._prepare_samples(root)
        
        # Using annotations from original dataset; these do not necessarily cover all races and genders
        if self.target == 'gender':
            self.classes = ['Male', 'Female']
        elif self.target == 'race':
            self.classes = ['White', 'Black', 'Asian', 'Indian', 'Others']
        else:
            raise ValueError(f"target {target} not recognized, must be either 'gender' or 'race'.")

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = pil_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        if self.target is not None:
            label = label[self.target]

        return image, label

    def __len__(self):
        return len(self.samples)

    def _prepare_samples(self, root):
        samples = []

        paths = glob.glob(os.path.join(root, '*'))

        for path in paths:
            if path.split('.')[0][-2:] == '-1':
                try:
                    label = self._load_label(path)
                except Exception as e:
                    # print('path: {}, exception: {}'.format(path, e))
                    continue

                samples.append((path, label))

        N = len(samples)
        if self.split == 'train':
            samples = samples[:int(0.9*N)]
        elif self.split == 'test':
            samples = samples[int(0.9*N):]

        return samples

    def _load_label(self, path):
        str_list = os.path.basename(path).split('.')[0].strip().split('_')
        age, gender, race = map(int, str_list[:3])
        if gender > 1 or gender < 0 or race > 4 or race < 0:
            raise Exception(f'Race or Gender parsed incorrectly, Race was {race} and Gender was {gender}')
        label = dict(age=age, gender=gender, race=race)
        return label

    def _load_datetime(self, s):
        return datetime.strptime(s, '%Y%m%d%H%M%S%f')