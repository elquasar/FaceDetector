# -*- coding: utf-8 -*-
import joblib
from sklearn.ensemble import RandomForestClassifier
from skimage import io
from skimage.feature import hog
import cv2
import time
import imutils

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image



#Import classifier from train.py

filename = 'classifier_model.sav'

clf = joblib.load(filename)

#Open image from train folder
image_id = 1
image = io.imread('test/%04d.jpg'%image_id)


#Define the windowSize : 
windowSize = (160,160)
#Define the scale for the pyramid function
scale = 1.5
#Define stepSize
stepSize = 10

image_ids = [i for i in range(1,501)]

#Creates detection dictionnary

detection = {}

for image_id in image_ids :  
    
    if image_id not in detection :
        detection[str(image_id)] = []
    
    image = io.imread('test/%04d.jpg'%image_id)
    

    for resized in pyramid(image, scale=1.5):
    
        for (x, y, window) in sliding_window(resized, stepSize, windowSize):
            
            if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[0] :
                continue
        
            clone = resized.copy()
            win_h = hog(window)
            prediction = clf.predict([win_h])
            if prediction == [1] : 
                
                detection[str(image_id)].append({
                        'id' : image_id,
                        'x' : x,
                        'y' : y,
                        'w' : window.shape[1],
                        'h' : window.shape[0],
                        'score' : clf.predict_proba([win_h])[0][1],
                        'window' : window
                        })
                
            # You can delete these comments to see what's happening : 
                
            
                cv2.rectangle(clone, (x, y), (x + windowSize[0], y + windowSize[0]), (0, 0, 255), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.05)
            else : 

                
                cv2.rectangle(clone, (x, y), (x + windowSize[0], y + windowSize[0]), (255, 0, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0)            
"""
# Generates detection.txt : 

output = ''

with open('detection.txt','w') as fichier: 
    
    for image_id in detection :
        
        for detections in detection[str(image_id)] : 
            
            output += str(round(detections['id'],2)) + ' ' + str(round(detections['y'],2)) + ' ' + str(round(detections['x'],2)) + ' '
            output += str(round(detections['h'],2)) + ' ' + str(round(detections['w'],2)) + ' ' + str(round(detections['score'],2))
            output += '\n'
                
    fichier.write(output)
                
"""
