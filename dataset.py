from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class FaceKeypointsDataset(Dataset):
    def __init__(self, dataframe, train=True, transform=None):
        self.dataframe = dataframe
        self.train = train
        self.transform = transform    
    def __len__(self):
        return len(self.dataframe)
    def __getitem__(self, idx):
        image = np.array(self.dataframe.iloc[idx,-1].split(" "),dtype="float32").reshape(96,96)
        if self.train:
            keypoints = self.dataframe.iloc[idx,0:30].values.astype(np.float32)
        else:
            keypoints = None
        sample = {'image': image, 'keypoints': keypoints}
        if self.transform:
            sample = self.transform(sample)    
        return sample