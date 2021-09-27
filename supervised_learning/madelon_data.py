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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

# DNN
dnn = MLPClassifier(hidden_layer_sizes=(1000), random_state=1,\
    learning_rate='invscaling', max_iter=40000, tol=0.00005, n_iter_no_change=100, verbose=VERBOSE)
dnn.fit(X_train, y_train)
print('\nPredict DNN:\n')
# print(dnn.predict(X_test))
print('Train data Score: ', dnn.score(X_train, y_train))
print('Test data Score', dnn.score(X_test, y_test))
plt = plot_learning_curve(dnn, 'DNN Learning Curve', X, y)
plt.savefig('madelon_dnn.png')

# KNN
knn = KNeighborsClassifier(n_neighbors=32, leaf_size=10, weights='distance', n_jobs=-1)
print('\nPredict KNN:')
knn.fit(X_train, y_train)
# print(knn.predict(X_test))
print('Train data Score: ', knn.score(X_train, y_train))
print('Test data Score', knn.score(X_test, y_test))
plt = plot_learning_curve(knn, 'KNN Learning Curve', X, y)
plt.savefig('madelon_knn.png')

# DecisionTree
print('\nDecision Trees:')
tree = DecisionTreeClassifier(random_state=0, min_samples_leaf=5)
tree.fit(X_train, y_train)
# print(tree.predict(X_test))
print('Train data Score: ', tree.score(X_train, y_train))
print('Test data Score', tree.score(X_test, y_test))
plt = plot_learning_curve(tree, 'Decision Trees Learning Curve', X, y)
plt.savefig('madelon_dt.png')

# Forest
print('\nRandom Forests:')
forest = RandomForestClassifier(random_state=0, min_samples_leaf=12)
forest.fit(X_train, y_train)
# print(forest.predict(X_test))
print('Train data Score: ', forest.score(X_train, y_train))
print('Test data Score', forest.score(X_test, y_test))
plt = plot_learning_curve(forest, 'Random Forests Learning Curve', X, y)
plt.savefig('madelon_forest.png')

# SVM
print('\nSVM:')
svms = svm.SVC()
svms.fit(X_train, y_train)
# print(svms.predict(X_test))
print('Train data Score: ', svms.score(X_train, y_train))
print('Test data Score', svms.score(X_test, y_test))
plt = plot_learning_curve(svms, 'SVM Learning Curve', X, y)
plt.savefig('madelon_svm.png')
