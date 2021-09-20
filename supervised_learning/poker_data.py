import pandas as pd

from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

VERBOSE = False

poker_train_df = pd.read_csv('../datasets/poker/poker-hand-training-true.data', sep=',')
poker_test_df = pd.read_csv('../datasets/poker/poker-hand-testing.data', sep=',')

X_train = poker_train_df.values[:, :-1]
y_train = poker_train_df.values[:,-1]

X_test = poker_test_df.values[:, :-1]
y_test = poker_test_df.values[:,-1]

# # DNN

# dnn = MLPClassifier(hidden_layer_sizes=(32,64,32), random_state=1, max_iter=10000, verbose=True).fit(X_train, y_train)
# dnn.predict_proba(X_test[:1])

# print(dnn.predict(X_test))
# print('Train data Score: ', dnn.score(X_train, y_train))
# print('Test data Score', dnn.score(X_test, y_test))


# # KNN
# knn = KNeighborsClassifier(n_neighbors=5, leaf_size=20, n_jobs=-1)
# print('\nPredict KNN:\n')
# knn.fit(X_train, y_train)
# print(knn.predict(X_test))
# print('Train data Score: ', knn.score(X_train, y_train))
# print('Test data Score', knn.score(X_test, y_test))


# # DecisionTree
# print('\nDecision Trees:\n')
# tree = DecisionTreeClassifier(random_state=0, min_samples_leaf=2, max_features=10)
# tree.fit(X_train, y_train)
# print(tree.predict(X_test))
# print('Train data Score: ', tree.score(X_train, y_train))
# print('Test data Score', tree.score(X_test, y_test))


# # Forest
# print('\nRandom Forests:\n')
# forest = RandomForestClassifier(random_state=0, max_features=2, min_samples_leaf=2)
# forest.fit(X_train, y_train)
# print(forest.predict(X_test))
# print('Train data Score: ', forest.score(X_train, y_train))
# print('Test data Score', forest.score(X_test, y_test))


# SVM
svms = svm.SVC()
svms.fit(X_train, y_train)
print(svms.predict(X_test))
print('Train data Score: ', svms.score(X_train, y_train))
print('Test data Score', svms.score(X_test, y_test))
