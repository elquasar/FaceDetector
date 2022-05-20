# Presentation of the project 

This project is a face detector made for the SY32 course at the UTC.

Important : to make the scripts work, it is necessary to install opencv and imutils here are the commands to enter in a python console :

pip install opencv-python pip install imutils

This folder contains 4 files :

* train.py : which is the training script of my face detector, this file generates a classifier model : classifier_model.sav which is used by test.py

* test.py : which is the file allowing to test the face detector. This file generates a detection.txt file which is the result of the detections

# How to use it :

* Place the project_train file containing the training images and the label_train file in the directory of this project. 
* Place the test file containing the images in the directory of this file.

Run the train.py script to generate the classifier model (optional, it is already in the directory), be careful, the generation can take a lot of time depending on your configuration!

Run the test.py script to generate the detection.txt file

(optional) If you want to see the algorithm working on an image visually, it is possible to remove the comments at the end of the test.py script. 
