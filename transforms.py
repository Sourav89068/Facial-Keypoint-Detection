from torchvision import transforms
import numpy as np
import torch

class Normalize():
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']        
        return {'image': image / 255.,'keypoints': keypoints}        

class ToTensor():
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        image = image.reshape(1,96,96)
        image = torch.from_numpy(image)        
        if keypoints is not None:
            keypoints = torch.from_numpy(keypoints)
            return {'image': image, 'keypoints': keypoints}
        else:
            return {'image': image}

class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, sample):        
        flip_indices = [(0, 2), (1, 3),
                        (4, 8), (5, 9), (6, 10), (7, 11),
                        (12, 16), (13, 17), (14, 18), (15, 19),
                        (22, 24), (23, 25)]        
        image, keypoints = sample['image'], sample['keypoints']        
        if np.random.random() < self.p:
            image = image[:, ::-1]
            if keypoints is not None:
                for a, b in flip_indices:
                    keypoints[a], keypoints[b]= keypoints[b], keypoints[a]
                keypoints[::2] = 96. - keypoints[::2]        
        return {'image': image,'keypoints': keypoints}