# IrisDataset_KNNClassifier
Application of KNN Classifer Algorithm on 'Iris' a popular dataset in Python
Dataset: The Iris dataset is a popular dataset that contains measurements of different features of iris flowers along with their species labels.

KNN Classifier: Using the KNN classifier with the Iris dataset involves training the model on the known features and labels of the iris flowers. When given new measurements of iris features, the KNN algorithm identifies the k closest data points in the training set and assigns the majority class among those neighbors as the predicted species for the new input.

Prediction Process: For instance, if you have new measurements of iris features like sepal length, sepal width, petal length, and petal width, the KNN classifier calculates distances between these features and the features of the training data. It then identifies the k closest data points and predicts the species label based on the majority class among these neighbors.

Hyperparameter: The key hyperparameter in KNN is the value of k, which determines the number of neighbors considered for classification. Choosing an appropriate value of k is important to balance between overfitting and underfitting.

