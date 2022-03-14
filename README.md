# wine label reader toolkit

> Computer vision project to read a label on a wine bottle using tensorflow, OpenCV and Tesseract.

The goal here is to make the computer read the label of a bottle of wine from a simple photo. Why is this complicated, you may ask? Well, first of all, we can't call directly an OCR (Optical Character Recognition) library like tesseract, because the text on the label is distorted on a cylinder and because of that, we can't extract correctly the characters and thus the words and sentences.

To use this package on your machine you have to install the dependencies in the requirement.txt move the photo you want to train in the "X" folder and the masks in the "y" folder and the photo you want to read in the "to_read" folder.
To configure the files location or the parameters of the U-net use the Config.json file.

## train model on images
```
python main.py --train
```
## read images
```
python main.py --read
```
## train then read images
```
python main.py --train --read
```

# Live demo online:
https://plural.run/wineReader
