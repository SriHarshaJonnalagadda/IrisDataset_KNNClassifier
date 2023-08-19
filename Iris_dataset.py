from sklearn.datasets import load_iris
import pandas as pd
iris_dataset = load_iris()
print("Keys of Dataset:",iris_dataset.keys())
print(iris_dataset['DESCR'][: 500]+"\n....")
print('Target Names:',iris_dataset['target_names'])
print("Feature Names:",iris_dataset['feature_names'])
print('Type of data:',type(iris_dataset['data']))
print('Shape of Data:',iris_dataset['data'].shape)
print('DATA:',iris_dataset['data'][:10])
print('Target:',iris_dataset['target'])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)
print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_test shape:',X_test.shape)
print('y_test shape:',y_test.shape) 
#create a dataframe from data in X_train--
#Label columns using the strings in iris_dataset.features_name---
iris_dataframe = pd.DataFrame(X_train,columns=iris_dataset.feature_names)
#create ascatter matrix from dataframe]
import matplotlib.pyplot as plt
import mglearn
pd.plotting.scatter_matrix(iris_dataframe,c =y_train,figsize=(10,10),marker = 'o',hist_kwds={'bins':20},s = 60,alpha=0.7,cmap = mglearn.cm3)
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train) #To build the model on a training set
#Making predictions to check if the trained model is working or not
#Giving sample input--
import numpy as np
X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)
print('Prediction:',prediction)
print('Predicted species:',iris_dataset['target_names'][prediction])
# Evaluating the model--
#To evaluate the model, predict the values of X_test
y_pred = knn.predict(X_test)
print('Test set Predictions:',y_pred)
print('Test set score:',knn.score(X_test,y_test))

#%%
