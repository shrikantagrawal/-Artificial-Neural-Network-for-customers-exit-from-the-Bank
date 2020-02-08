# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:01:55 2020

@author: Shrikant Agrawal
"""


import matplotlib.pyplot as plt   # For plotting diagrams
import numpy as np                # For creating arrays
import pandas as pd               # To read the dataset

# Import dataset
dataset= pd.read_csv('BankCustomers.csv')

# Divide the dataset between dependent ana indepnedent variable. Remove columns which are not required
X=dataset.iloc[:,3:13]          
y=dataset.iloc[:,13]

# Convert categorical variable into dummy variables
states=pd.get_dummies(X['Geography'],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

#Drop Gender and Geography column from the original dataset
X=X.drop(['Geography','Gender'],axis=1)

#Concatenate dummny columns
X=pd.concat([X,states,gender],axis=1)

#Split the datset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)       #fit method is done only once not twice hence we have not used fit here

# All steps are similare till here are similar like Machine Learning, only model development is different


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages. Alternatively we can do this operation using tenser flow
import keras
from keras.models import Sequential    # It helps to create sequential neural network
from keras.layers import Dense         # It helps to create hidden layers

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
classifier.summary()

# Compiling the ANN. We are using loss ='Binary_crossentropy because we have two outputs
#for more than 2 variables use categorical crossentropy
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# When you run above command loss ie error rate gets reduce and accuracy percentate gets increase


# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)