import cv2
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor


class MaskDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataFrame = dataFrame
        
        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(), # [0, 1]
        ])
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            raise NotImplementedError('slicing is not supported')
        
        row = self.dataFrame.iloc[key]
        image_ = row['image']
        imread = cv2.imread(image_)
        NoneType = type(None)
        if type(imread) == NoneType:
            print(image_)
            # import os
            # os.remove(image_)
        else:
            return {
                'image': self.transformations(imread),
                'mask': tensor([row['mask']], dtype=long)
            }
    
    def __len__(self):
        return len(self.dataFrame.index)
