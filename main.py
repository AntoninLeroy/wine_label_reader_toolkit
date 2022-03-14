import sys
from wineReader.utils import *
from wineReader.model import *
from wineReader.labelVision import *

# run with argument
# --train
# --readLabel

f = open('Config.json')
Config = json.load(f)

job = sys.argv
    
if len(job) <= 1:
    print("\n Please provide one or both arguments: \n --train \n --read \n")
    quit()

if "--train" in job:
    # clean training & validation folders
    clean_training_folders(Config)

    # load datas
    X_train,X_valid,y_train,y_valid = load_train_valid_split(Config)

    # train u-net model
    unet = Unet()
    unet.fit(X_train,X_valid,y_train,y_valid,Config)

if "--read" in job:
    # clean results folder
    clean_results_folder(Config)

    # load source img to read and unet inputs
    X, srcs, fileNames = load_label_to_read(Config)

    # load trained model
    model = keras.models.load_model(Config['model_to_predict_path'])

    # get U-net label predictions
    unet = Unet()
    unet_output = unet.predict(X, model, fileNames, Config)

    # read labels
    label = labelVision(Config)
    label.readLabels(unet_output, srcs, fileNames)
    