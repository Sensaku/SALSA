#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 19:07:43 2023

@author: cecile capponi
"""
import os

import matplotlib.pyplot as plt
from skimage import exposure
from skimage.color import rgb2gray
from PIL import Image
from numpy import asarray
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

"""
Computes a representation of an image from the (gif, png, jpg...) file 
representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels 
other to be defined
input = an image (jpg, png, gif)
output = a new representation of the image
"""   

def raw_image_to_representation(image, representation):
    if representation == "HC":
        opened_image = plt.imread(image)
        colors = ['red', 'green', 'blue']
        result = []
        for index, color in enumerate(colors):
            hist, bins = exposure.histogram(opened_image[:, :, index], nbins=256)
            if (index == 0):
                result.append(bins)
            result.append(hist)
        return result
    
    if representation == "PX":
        return asarray(Image.open(image))

    if representation == "GC":
        return rgb2gray(plt.imread(image)) 

"""
Returns a data structure embedding train images described according to the 
specified representation and associate each image to its label.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been transformed and labelled according to the directory they are
stored in.
-- uses function raw_image_to_representation
"""

def get_label(category):
    if (category == "Mer"):
        return 1
    else:
        return -1

def load_transform_label_train_data(directory, representation):
    data = {"representations":[], "labels":[], "filenames":[]}
    for folder in os.listdir(directory):
        for filename in os.listdir(os.path.join(directory, folder)):
            print(os.path.join(directory, folder, filename))
            data["representations"].append(raw_image_to_representation(os.path.join(directory, folder, filename), representation))
            data["labels"].append(get_label(folder))
            data["filenames"].append(filename)
    return data
    
    
"""
Returns a data structure embedding test images described according to the 
specified representation.
-> Representation can be (to extend) 
'HC': color histogram
'PX': tensor of pixels 
'GC': matrix of gray pixels 
other to be defined
input = where are the data, which represenation of the data must be produced ? 
output = a structure (dictionnary ? Matrix ? File ?) where the images of the
directory have been transformed (but not labelled)
-- uses function raw_image_to_representation
"""

def load_transform_test_data(directory, representation):
    data = {"representations":[], "labels":[], "filenames":[]}
    for folder in os.listdir(directory):
        for filename in os.listdir(directory):
            data["representations"].append(raw_image_to_representation(os.path.join(directory, filename), representation))
            data["labels"].append(filename)
    return data

"""
Learn a model (function) from a representation of data, using the algorithm 
and its hyper-parameters described in algo_dico
Here data has been previously transformed to the representation used to learn
the model
input = transformed labelled data, the used learning algo and its hyper-parameters (a dico ?)
output =  a model fit with data
"""

def learn_model_from_data(train_data, algo_dico):
    data = train_data["representations"]
    target = train_data["labels"]
    model = GaussianNB()
    model.fit(data, target)
    return model

"""
Given one example (representation of an image as used to compute the model),
computes its class according to a previously learned model.
Here data has been previously transformed to the representation used to learn
the model
input = representation of one data, the learned model
output = the label of that one data (+1 or -1)
-- uses the model learned by function learn_model_from_data
"""

def predict_example_label(example, model):
    label = model.predict([example])[0]
    return label


"""
Computes an array (or list or dico or whatever) that associates a prediction 
to each example (image) of the data, using a previously learned model. 
Here data has been previously transformed to the representation used to learn
the model
input = a structure (dico, matrix, ...) embedding all transformed data to a representation, and a model
output =  a structure that associates a label to each data (image) of the input sample
"""

def predict_sample_label(data, model):
    predictions = model.predict(data)
    return predictions

"""
Save the predictions on data to a text file with syntax:
filename <space> label (either -1 or 1)  
NO ACCENT  
Here data has been previously transformed to the representation used to learn
the model
input = where to save the predictions, structure embedding the data, the model used
for predictions
output =  OK if the file has been saved, not OK if not
"""

def write_predictions(directory, filename, data, model):
    try:
        file = open(os.path.join(directory, filename), 'x')
    except FileExistsError:
        return "not OK"
    predictions = predict_sample_label(data["representations"], model)
    for index in range(0, len(data)):
        file.write(data["filenames"][index] + " label " + str(predictions[index]) + "\n")
    return "OK"

"""
Estimates the accuracy of a previously learned model using train data, 
either through CV or mean hold-out, with k folds.
Here data has been previously transformed to the representation used to learn
the model
input = the train labelled data as previously structured, the learned model, and
the number of split to be used either in a hold-out or by cross-validation  
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random
"""

def estimate_model_score(train_data, algo_dico, k):
    X_train, X_test, y_train, y_test = train_test_split(train_data["representations"], train_data["labels"], test_size=k)
    model = learn_model_from_data({"representations":X_train, "labels":y_train}, algo_dico)
    y_predits = predict_sample_label(X_test, model)
    return accuracy_score(y_test, y_predits)
