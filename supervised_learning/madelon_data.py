import pandas as pd

from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

VERBOSE=False

madelon = pd.read_csv('../datasets/madelon/madelon.csv', sep=',')

X = madelon.values[:, :-1]
y = madelon.values[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# DNN

# dnn = MLPClassifier(hidden_layer_sizes=(12,), random_state=1, max_iter=10000, tol=0.0001, n_iter_no_change=100, verbose=VERBOSE)
# dnn.fit(X_train, y_train)
# print('\nPredict DNN:\n')
# # print(dnn.predict(X_test))
# print('Train data Score: ', dnn.score(X_train, y_train))
# print('Test data Score', dnn.score(X_test, y_test))

# KNN
knn = KNeighborsClassifier(n_neighbors=30, leaf_size=10, weights='distance', n_jobs=-1)
print('\nPredict KNN:')
knn.fit(X_train, y_train)
# print(knn.predict(X_test))
print('Train data Score: ', knn.score(X_train, y_train))
print('Test data Score', knn.score(X_test, y_test))

# DecisionTree
print('\nDecision Trees:')
tree = DecisionTreeClassifier(random_state=0, min_samples_leaf=5)
tree.fit(X_train, y_train)
# print(tree.predict(X_test))
print('Train data Score: ', tree.score(X_train, y_train))
print('Test data Score', tree.score(X_test, y_test))



# Forest
print('\nRandom Forests:')
forest = RandomForestClassifier(random_state=0, min_samples_leaf=12)
forest.fit(X_train, y_train)
# print(forest.predict(X_test))
print('Train data Score: ', forest.score(X_train, y_train))
print('Test data Score', forest.score(X_test, y_test))


# SVM
print('\nSVM:')
svms = svm.SVC()
svms.fit(X_train, y_train)
# print(svms.predict(X_test))
print('Train data Score: ', svms.score(X_train, y_train))
print('Test data Score', svms.score(X_test, y_test))
