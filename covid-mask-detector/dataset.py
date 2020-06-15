import cv2
from torch import long, tensor
from torch.utils.data.dataset import Dataset


class MaskDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image']
        img = cv2.imread(img_path)
        if img is not None:
            return {
                'image': self.transform(img),
                'mask': tensor([row['mask']], dtype=long)
            }
    
    def __len__(self):
        return len(self.dataframe)
