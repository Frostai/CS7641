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

# DNN
dnn = MLPClassifier(hidden_layer_sizes=(100,100), random_state=1, max_iter=10000, tol=0.0001, n_iter_no_change=10, verbose=VERBOSE)
dnn.fit(X_train, y_train)
print('\nPredict DNN:\n')
# print(dnn.predict(X_test))
print('Train data Score: ', dnn.score(X_train, y_train))
print('Test data Score', dnn.score(X_test, y_test))
plt = plot_learning_curve(dnn, 'DNN Learning Curve', X_train, y_train, n_jobs=-1)
plt.savefig('poker_dnn.png')

# KNN
knn = KNeighborsClassifier(n_neighbors=20, p=2, leaf_size=10, weights='distance', n_jobs=-1)
print('\nPredict KNN:')
knn.fit(X_train, y_train)
# print(knn.predict(X_test))
print('Train data Score: ', knn.score(X_train, y_train))
print('Test data Score', knn.score(X_test, y_test))
plt = plot_learning_curve(knn, 'KNN Learning Curve', X_train, y_train, n_jobs=-1)
plt.savefig('poker_knn.png')

# DecisionTree
print('\nDecision Trees:')
tree = DecisionTreeClassifier(random_state=0, min_samples_leaf=5)
tree.fit(X_train, y_train)
# print(tree.predict(X_test))
print('Train data Score: ', tree.score(X_train, y_train))
print('Test data Score', tree.score(X_test, y_test))
plt = plot_learning_curve(tree, 'Decision Tree Learning Curve', X_train, y_train, n_jobs=-1)
plt.savefig('poker_dt.png')


# Forest
print('\nRandom Forests:')
forest = RandomForestClassifier(random_state=0, min_samples_leaf=12)
forest.fit(X_train, y_train)
# print(forest.predict(X_test))
print('Train data Score: ', forest.score(X_train, y_train))
print('Test data Score', forest.score(X_test, y_test))
plt = plot_learning_curve(forest, 'Random Forest Learning Curve', X_train, y_train, n_jobs=-1)
plt.savefig('poker_forest.png')

# SVM
import time
print('\nSVM:')
svms = svm.SVC()
svms.fit(X_train, y_train)
# print(svms.predict(X_test))
print('Train data Score: ', svms.score(X_train, y_train), time.time() / 1000)
print('Test data Score', svms.score(X_test, y_test), time.time() / 1000)
plt = plot_learning_curve(svms, 'SVM Learning Curve', X_train, y_train, n_jobs=-1)
plt.savefig('poker_svm.png')