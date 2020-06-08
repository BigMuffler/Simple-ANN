# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#ANN
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values #independent variable
y = dataset.iloc[:, 13].values #dependent variables

#Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
le = LabelEncoder() #encoding gender column
X[:,2] = le.fit_transform(X[:,2])
#encoding countries, no order relationships between countries
ct1 = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[1])], remainder = 'passthrough')
X = np.array(ct1.fit_transform(X))# each new column will represent a categorical data


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
# Create ANN
# Importing Keras Library and Packages
import keras 
from keras.models import Sequential
from keras.layers import Dense

# Initializing ANN
ann = Sequential() 

# Adding the input layer and first hidden layer
ann.add(Dense(units = 6, activation = 'relu')) #nodes in hidden layer equals average of outputs + inputs

# Adding second hidden layer
ann.add(Dense(units = 6, activation = 'relu'))

# Adding Output Layer
ann.add(Dense(units = 1, activation = 'sigmoid'))

# Training the ANN
# binary_crossentropy for binary outcomes
# 'adam' for stochastic gradient descent
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Making Predictions
#Any input of predict must be a 2-D array
#Outputs probability of customer leaving or not
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5)

#Predicting Test Set Results
#[0 0] first column indicates if user actually stayed or left in the bank. Second Column represents prediction
y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Create Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test, y_pred)
