# Facial-Keypoint-Detection

## **About the data**

The data is taken from one of the kaggle competitions [dataset_link](https://www.kaggle.com/c/facial-keypoints-detection/data.)



## **Setup**
Download the data from the above link and keep extract training.csv and test.csv in one single directory.
Then fork all the .py and .ipynb files and keep all of them into the same directory where the data is.

## **Training**
Run the training.ipynb to start the training.(all data files .py files should be under same directory, otherwise give the path accordingly)

## Trained model
Download the trained model from the google drive [model_link](https://drive.google.com/file/d/1iKpoxyaQ7QX6zCxukkX7evi2UNH96YwC/view?usp=sharing)

Note: This model is necessary for testing.ipynb and deploy folder. Keep the model in both place. 

## **Local Deployment**
Copy the model in deploy folder and then

Run the file main.py from the deploy directory by the command

```
  python3 main.py
```
it will start the backend server. Then double click on the index.html to start the local webpage. One example image(RGB) is given in the deployment directory to see the deployment.

# *THE MODEL*

### References:
1. https://ieeexplore.ieee.org/document/9065279

2. https://arxiv.org/abs/1710.05279

3. https://towardsdatascience.com/facial-keypoints-detection-image-and-keypoints-augmentation-6c2ea824a59

4. https://github.com/yinguobing/cnn-facial-landmark

5. https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/

6. https://medium.com/diving-in-deep/facial-keypoints-detection-with-pytorch-86bac79141e4
