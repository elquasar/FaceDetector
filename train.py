
#=================================LIBRAIRIES==================================#

from skimage.feature import hog
import numpy as np
from skimage import io
from numpy import random
import imutils
import cv2
import time
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#==================================FONCTIONS==================================#

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def IoU(r1, r2):
    
    # r1 dict : 
    
    #      'x' = position x of the left top corner 
    #      'y' = position y of the left top corner
    #      'w' = width of the window
    #      'h' = height of the window
    
    dx = min(r1['x'] + r1['w'], r2['x'] + r2['w']) - max(r1['x'], r2['x'])
    dy = min(r1['y'], r2['y']) - max(r1['y'] - r1['h'], r2['y'] - r2['h'])
    
    
    if (dx >= 0) and (dy >= 0): 
        overlap = dx * dy
    else :
        overlap = 0

    a_r1 = r1['w'] * r1['h']
    a_r2 = r2['w'] * r2['h']
    
    union = np.absolute(float(a_r1 + a_r2 - overlap))
    
    IoU = overlap / union
    return IoU


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def parse_trainer(trainer_file):
    results = {}
    lines = []
    with open(trainer_file, 'r') as fp:
        lines = fp.read().splitlines()
        fp.close()

    for line in lines:
        
        data = [int(x) for x in line.split('\x20')]
        
        image_id = str(data[0])
        if image_id not in results:
            results[image_id] = []

        results[image_id].append({
           'x': data[2],
           'y': data[1],
           'h': data[3],
           'w': data[4]
        })
    return results

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def sliding_window(image, stepSize, windowSize):

    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):

            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def pyramid(image, scale=1.5, minSize=(30, 30)):

    yield image

    while True:

        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#        
def winSet(image, scale, stepSize,windowSize,nbr=30) : 
    
    winSet_f = {}
    winSet = {}
    window_id = 1
    
    
    # Creates a lot of window in the image
    
    for resized in pyramid(image, scale=1.5):
        
        for (x, y, window) in sliding_window(resized, stepSize, windowSize):
           
            if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[0] :
                continue

            winSet_f[str(window_id)] = {
                'x' : x,
                'y' : y,
                'h' : window.shape[0],
                'w' : window.shape[1],
                'window' : window
            }
            window_id += 1

    
    # Picks randomely  windows from this set : 
    
    for i in range(nbr): 
        
        r = random.randint(1, len(winSet_f))
        
        
        winSet[str(r)] = winSet_f[str(r)]

    
    return winSet
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def clean(windows,trainer,image_id) : 
    
    real_faces = trainer[str(image_id)]
    win_ids    = [identified for identified in windows]
    

    
    for face in real_faces : 
        
        for win_id in win_ids : 
            
            if IoU(face,windows[win_id]) < 0.5 : 
                
                windows[win_id].update({'status' : 'false'})
            else : 
                windows[win_id].update({'status' : 'true'})
                
    return windows
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def show_face(image,trainer, image_id,windowSize) :
    
    train = trainer[str(image_id)]
    
    show_face = train
    
    for j,face in enumerate(train) : 
        
        y = int(face['y'])
        x = int(face['x'])
        h = windowSize[0]
        w = windowSize[1]
        
        

        show_face[j]['window'] = image[y:y+h,x:x+w]
        show_face[j]['status'] = 'true'
        
        if show_face[j]['window'].shape[0] != windowSize[0] : 
            show_face[j]['window'] = show_face[j]['window'][0:windowSize[0],0:]
        elif show_face[j]['window'].shape[0] != windowSize[0] :    
            show_face[j]['window'] = show_face[j]['window'][0:,0:windowSize[0]]

    return show_face

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def create_x_y_train(true_faces,cleaned_Win) : 
    
    #Creating x_train, y_train
    
    x_train = []
    
    y_train = []
    
    #Puts into x_train the set of data corresponding to clean_windows['window'] 
    #+ hog them 
    
    for window_id in cleaned_Win : 
        

        if cleaned_Win[str(window_id)]['status'] == 'false' :
            
            process = hog(cleaned_Win[window_id]['window'])
            x_train.append(process) 
            y_train.append(-1)


    for faces in true_faces : 
    
        processf  = hog(faces['window']).tolist() 
        
        i = len(processf)
        h = 26244 - i

        while h !=0 : 
            processf.append(0)
            h-=1
        x_train.append(processf)
        y_train.append(+1)

    
    return (x_train,y_train)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

#=================================SCRIPT======================================#


#Creates x_train,y_train

x_train = []
y_train = []


#Define the windowSize : 
windowSize = (160,160)
#Define the scale for the pyramid function
scale = 1.5
#Define stepSize
stepSize = 180
#Imports trainer and parses it
trainer = parse_trainer('project_train/label_train.txt')


image_ids = [i for i in range(1,1001)]
j = 0
 
for image_id in image_ids :
        
    #Imports image from train
    image = io.imread('project_train/train/%04d.jpg'%image_id)
    #Creates a set of window in this image
    windows = winSet(image, scale, stepSize,windowSize)
    #Cleans these windows using trainer
    cleaned = clean(windows,trainer,image_id)
    #Finds real faces in the current image
    faces = show_face(image,trainer, image_id,windowSize)
    #Creates x_train and y_train for the current image
    x_y_train_cur = create_x_y_train(faces,cleaned)
    #Adds into the bigger x_train,y_train, datas of the current image
    x_train.append(x_y_train_cur[0])
    y_train.append(x_y_train_cur[1])
    j+=1
    print('Genereting training datas : ',j/10,'%\r')

y_train = [item for sublist in y_train for item in sublist]
x_train = [item for sublist in x_train for item in sublist]


#1st learning : 

print('Starting 1st learning....')

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)

print('Finished')

# 2nd learning : 

#Defines stepSize for sliding window
stepSize = 40

#list for all false_positive we find
false_positive = []


print('2nd learning :')
print('Hard-negative mining ....')

j = 0

# Using the classifier on every image of the sliding-window algorithm :

for image_id in image_ids :
    
    image = io.imread('project_train/train/%04d.jpg'%image_id)
    
    for resized in pyramid(image, scale=1.5):
        
        for (x, y, window) in sliding_window(resized, stepSize, windowSize=(160,160)):
           
            if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[0] :
                continue

            current_win = {
                    'x' : x,
                    'y' : y,
                    'w' : resized.shape[1],
                    'h' : resized.shape[0],
                    'window' : hog(window)
                        
                    }
               
            prediction = rfc.predict([current_win['window']])   

            if prediction == [1] : 
                
                for faces in trainer[str(image_id)] : 
                    if IoU(current_win,faces) < 0.5 :
                        false_positive.append(current_win)
    j+=1
    print('Applying classifier : ',j/10,'%','\r')

print('Starting 2nd learning :')

#Selects 100 windows from the set of false-positive windows (in order to avoid overfitting)

for i in range(100) :
    
    x_train.append(random.choice(false_positive)['window'])
    y_train.append(-1)



# Searching for the best paramaters of our RandomForestClassifier : 


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



#Now we can test combinations of these parameters in a random way :



rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = param_grid, 
                               n_iter = 3, cv = 3, verbose=2, random_state=42, n_jobs = -1)


rf_random.fit(x_train,y_train)

# Takes the best classifier we just found

best_rfc = rf_random.best_estimator_

#Saves it, in a .sav file

filename = 'classifier_model.sav'
joblib.dump(best_rfc, filename)


print('Classifier successfully trained, a file containing classifier\'s configuration has been created !')



