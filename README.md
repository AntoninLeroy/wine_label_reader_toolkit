# wine label reader toolkit

> Computer vision project to read a label on a wine bottle using tensorflow, OpenCV and Tesseract.

The goal here is to make the computer read the label of a bottle of wine from a simple photo. Why is this complicated, you may ask? Well, first of all, we can't call directly an OCR (Optical Character Recognition) library like tesseract, because the text on the label is distorted on a cylinder and because of that, we can't extract correctly the characters and thus the words and sentences.

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
