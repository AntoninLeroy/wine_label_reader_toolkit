import json
import shutil
from tqdm import tqdm
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request

def load_train_valid_split(Config):

        X = []
        y = []

        for file in tqdm(os.listdir(Config['X_path'])):

            img=cv2.imread(Config['X_path']+file)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(256,256))
            X.append(img)
            
            img=cv2.imread(Config['y_path']+file)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=cv2.resize(img,(256,256))
            y.append(list(img))

        X=np.array(X)
        y=np.array(y,dtype=np.bool)

        X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=Config['valid_size'])

        # Put img used for training in their respective folders
        for i, img in enumerate(X_train):
            cv2.imwrite(Config['X_train_path']+"{}.jpg".format(i), img)

        for i, img in enumerate(X_valid):
            cv2.imwrite(Config['X_valid_path']+"{}.jpg".format(i), img)

        return X_train,X_valid,y_train,y_valid

def clean_training_folders(Config):

    for file in os.listdir(Config['X_train_path']):
        os.remove(Config['X_train_path']+file)
    for file in os.listdir(Config['X_valid_path']):
        os.remove(Config['X_valid_path']+file)

def clean_results_folder(Config):

    # rm results folder and all folder / files in it
    shutil.rmtree(r"{}".format(Config['results_path']))
    # make a new one
    os.mkdir(Config['results_path'])

def load_label_to_read(Config):

        fileNames = []
        srcs = []
        X = []

        for file in tqdm(os.listdir(Config['to_read_path'])):

            #buid a file structure for results
            filename = file.rsplit( ".", 1 )[0]
            fileNames.append(filename)
            parent_dir = Config['results_path']
            path = os.path.join(parent_dir, filename)
            os.mkdir(path)

            src=cv2.imread(Config['to_read_path']+file)
            srcs.append(src)
            img=cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(256,256))
            X.append(img)

            cv2.imwrite(path + "/" + "0_src.jpg", src)
            cv2.imwrite(path + "/" + "1_unet.jpg", img)

        X=np.array(X)

        return X, srcs, fileNames

def img_url_to_input_unet(url):
    
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    X=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    X=cv2.resize(X,(256,256))

    return np.array([X]), img