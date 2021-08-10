import numpy as np
import paddle
from paddle.io import Dataset
import paddle.vision.transforms as T
class CifarDataset(Dataset):
    def __init__(self,dataset,dev_ratio,is_train=True,transform=None):
        assert dev_ratio>0 and dev_ratio<1
        train_num=int((1-dev_ratio)*len(dataset))
        self.train_set, self.dev_set = paddle.io.random_split(dataset, [train_num, len(dataset)-train_num])
        self.mode='train' if is_train else 'dev'
        self.transform=transform

    def __getitem__(self, idx):
        if self.mode=='train':
            img,label=self.train_set[idx]
            if self.transform is not None:img=self.transform(img)
            return img,label
        else:
            img,label=self.dev_set[idx]
            if self.transform is not None:img=self.transform(img)
            return img,label

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_set)
        else:
            return len(self.dev_set)

