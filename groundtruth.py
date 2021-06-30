import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_images_with_keypoints(image_set,indices,figsize=(20,10),ncolumns=5,show_keypoints=True):
    '''
    image_set: a dataframe containing the images
    indices: takes images numbers
    figsize: size of the figure
    ncolumns: number of columns in the figure 
    show_keypoints: boolean type(wheather we will see keypionts or not)
    '''
    nrows=np.int(np.ceil(len(indices)/ncolumns))
    plt.figure(figsize=figsize)
    i=1
    if show_keypoints:
        for image_no in indices:
            plt.subplot(nrows,ncolumns,i)
            plt.imshow(np.array(image_set["Image"][image_no].split(" ")).astype("uint8").reshape(96,96),cmap="gray")
            plt.scatter(image_set.iloc[image_no,0:30:2],image_set.iloc[image_no,1:30:2],s=20,marker="o",facecolors="m",edgecolors="b")
            plt.xticks([])
            plt.yticks([])
            i+=1
        plt.title("Ground Truths",fontdict={'family': 'serif','color': 'darkmagenta','weight': 'normal','size': 30,})
        
    else:
        for image_no in indices:
            plt.subplot(nrows,ncolumns,i)
            plt.imshow(np.array(image_set["Image"][image_no].split(" ")).astype("uint8").reshape(96,96),cmap="gray")
            plt.xticks([])
            plt.yticks([])
            i+=1
        plt.title("Test Images",fontdict={'family': 'serif','color': 'darkmagenta','weight': 'normal','size': 30,})