from flask import Flask,jsonify,make_response
from flask_cors import CORS
from imagepredkey import CNN
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)
@app.route("/")
def index():
    return "Congratulations, it's a web app!"

@app.route("/<photo>")
def img(photo):
    model = CNN(outputs=30)        
    model.load_state_dict(torch.load('aug_cnn.pt'))

    a=cv2.imread(photo)
    a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
    a=cv2.resize(a/255,(96,96))
    a=np.expand_dims(a,0)
    a=np.expand_dims(a,0)
    im = Variable(torch.from_numpy(a))
    im=im.type(torch.FloatTensor)
    model.eval()
    with torch.no_grad():
    	keypoints=model(im)
    keypoints=keypoints.data.numpy()
    tstim_first=cv2.imread(photo)
    dim=96,96
    tstim1=cv2.resize(tstim_first[:,:,0],dim, interpolation = cv2.INTER_AREA)
    tstim2=cv2.resize(tstim_first[:,:,1],dim, interpolation = cv2.INTER_AREA)
    tstim3=cv2.resize(tstim_first[:,:,2],dim, interpolation = cv2.INTER_AREA)
    tstim=np.zeros((96,96,3))
    tstim[:,:,0]=tstim1
    tstim[:,:,1]=tstim2
    tstim[:,:,2]=tstim3
    plt.imshow(tstim.astype("int64")[:,:,::-1])
    plt.scatter(keypoints[0][0::2],keypoints[0][1::2],s=20,marker ='o', c='r',edgecolors='y')
    plt.xticks([])
    plt.yticks([])
    plt.savefig("a.png")

    res= make_response(jsonify({"filename":"a.png"}),200) 
    return res

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8081, debug=True)
