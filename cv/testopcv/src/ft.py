# this file is for the training 
import cv2
import os
import numpy as np
from PIL import Image
import pickle
 
face_cascade = cv2.CascadeClassifier('C:/Users/bouji/cv/testopcv/src/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

#the absulute path for the image diractory 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#the name of the folder 
image_dir = os.path.join(BASE_DIR, "images")

current_id = 0
label_ids = {}

# the array for the paths and the libles
y_labels = []
x_train = []

# fetch in the image diractory for all  the directories and then on the pictures with jpg and png extention
for root, dirs, files in os.walk(image_dir):# fetch in the image diractory for all  the directories 
    for file in files: # fetch  on all the pictures with jpg and png extention
        if file.endswith("png") or file.endswith("jpg"):
            
            path = os.path.join(root, file) #contact and create the new path
            label = os.path.basename(root).replace(" ", "-").lower() #replace space with -
            
            
            if not label in label_ids:  #affect ids for each person or label
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
           
    
            pil_image = Image.open(path).convert("L") # convert image to grayscale (vecror < type,mode,size, @ >)
            #print(pil_image)
            size = (550, 550)
            final_image = pil_image.resize(size, Image.ANTIALIAS) #resize
            
            image_array = np.array(final_image, "uint8") # our image is a matrice
            #print(image_array)
            
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) #detect  scale 
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(x_train) 
print(label_ids)
#print(y_labels)


with open("pickles/face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizers/face-trainner.yml")