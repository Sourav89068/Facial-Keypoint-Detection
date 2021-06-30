import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from model import CNN
model=CNN(outputs=30)
model.load_state_dict(torch.load('CNN.pt'))

def predict(image):
    '''
    Takes an image(RGB) with full path as an input
    '''
    im=cv2.imread(image)
    im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    image_copy=im.copy()
    im=im/255
    im=np.expand_dims(im,0)
    im=np.expand_dims(im,0)
    im=Variable(torch.from_numpy(im))
    im=im.type(torch.FloatTensor)
    model.eval()
    with torch.no_grad():
        keypoints=model(im)
    keypoints=keypoints.data.numpy()
    plt.imshow(image_copy,cmap='gray')
    plt.scatter(keypoints[0][0::2],keypoints[0][1::2],s=5,edgecolors='r',c='y')
    plt.show()
    return

def predict_on_test_set_images(image_set,indices,figsize=(20,10),ncolumns=5):
    nrows=np.int(np.ceil(len(indices)/ncolumns))
    plt.figure(figsize=figsize)
    i=1
    for image_no in indices:
        im=np.array(image_set["Image"][image_no].split(" "),dtype="uint8").reshape(96,96)
        image_copy=im.copy()
        im=im/255
        im=np.expand_dims(im,0)
        im=np.expand_dims(im,0)
        im=Variable(torch.from_numpy(im))
        im=im.type(torch.FloatTensor)
        model.eval()
        with torch.no_grad():
            keypoints=model(im)
        keypoints=keypoints.data.numpy()
        plt.subplot(nrows,ncolumns,i)
        plt.imshow(image_copy,cmap='gray')
        plt.scatter(keypoints[0][0::2],keypoints[0][1::2],s=5,edgecolors='r',c='y')
        plt.xticks([])
        plt.yticks([])
        i+=1
    return