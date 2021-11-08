import time

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import FastICA, PCA, KernelPCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from plotting import plot_nn_curve_df, plot_learning_curve
from clustering import em, kmeans

VERBOSE=False
SEED = 1337

# MADELON dataset
madelon = pd.read_csv('../datasets/madelon/madelon.csv', sep=',')
X_madelon = madelon.values[:, :-1]
y_madelon = madelon.values[:,-1]
X_train_madelon, X_test_madelon, y_train_madelon, y_test_madelon = train_test_split(X_madelon, y_madelon, test_size=0.4, random_state=1)

# Credit dataset
encoder = OneHotEncoder(handle_unknown='ignore')
credit = pd.read_csv('../datasets/credit/dataset_31_credit-g.csv', sep=',')
encoder.fit(credit)
credit_t = encoder.transform(credit).toarray()
X_credit = credit_t[:, :-1]
y_credit = credit_t[:,-1]
X_train_credit, X_test_credit, y_train_credit, y_test_credit = train_test_split(X_credit, y_credit, test_size=0.4, random_state=1)


def ica(n, X):
    ica = FastICA(n_components=n, max_iter=10000, )
    ica.fit(X)
    return ica.transform(X)

def pca(n, X):
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca.transform(X)

def grp(n, X):
    grp = GaussianRandomProjection(n_components=n)
    return grp.fit_transform(X)

def kpca(n, X):
    kpca = KernelPCA(n_components=n)
    return kpca.fit_transform(X)


def dnn(X, y, layer_sizes=(100), title='', fileprefix='', saveplots=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
    dnn = MLPClassifier(hidden_layer_sizes=layer_sizes, random_state=1,\
    learning_rate='constant', max_iter=25000, tol=0.00005, n_iter_no_change=100, verbose=VERBOSE)
    if saveplots:
        plots = plot_learning_curve(dnn, title, X, y, n_jobs=-1)
        plots.savefig('figures/{}_dnn.png'.format(fileprefix))
    dnn.fit(X_train, y_train)
    print('\nPredict DNN {} {}:\n'.format(title, fileprefix))
    print('Train data Score: ', dnn.score(X_train, y_train))
    print('Test data Score', dnn.score(X_test, y_test))
    

# debug
def test_pca():
    for i in range(1,40):
        transformed, explained_variance = pca(i, madelon)
        print(explained_variance)

def run_dnn_experiments(plots=True):
    # Run DNN on MADELON reduced
    # Run PCA with 5 and 8 components
    dnn(ica(5, X_madelon), y_madelon, title='MADELON ICA (n=5)', saveplots=plots, fileprefix='ica_n5')
    dnn(pca(5, X_madelon), y_madelon, title='MADELON PCA (n=5)', saveplots=plots, fileprefix='pca_n5')
    dnn(pca(8, X_madelon), y_madelon, title='MADELON PCA (n=8)', saveplots=plots, fileprefix='pca_n8')
    dnn(grp(8, X_madelon), y_madelon, title='MADELON GRP (n=8)', saveplots=plots, fileprefix='grp_n8')
    dnn(kpca(8, X_madelon), y_madelon, title='MADELON KPCA (n=8)', saveplots=plots, fileprefix='kpca_n8')

    # Run DNN on Credit dataset reduced
    dnn(ica(5, X_credit), y_credit, title='CREDIT ICA (n=5)', saveplots=plots, fileprefix='ica_n5')
    dnn(pca(5, X_credit), y_credit, title='CREDIT PCA (n=5)', saveplots=plots, fileprefix='pca_n5')
    dnn(pca(8, X_credit), y_credit, title='CREDIT PCA (n=8)', saveplots=plots, fileprefix='pca_n8')

# Run ICA, PCA, GRP, and ___ and cluster after
def run_experiments():
    # ICA experiments
    ica_madelon = ica(5, X_madelon)
    ica_credit =  ica(5, X_credit)
    kmeans(50, X=ica_madelon, y=y_madelon, dataset='MADELON', filename_prefix="ica_madelon")
    kmeans(50, X=ica_credit, y=y_credit, dataset='Credit', filename_prefix="ica_credit")
    em(50, X=ica_madelon, dataset='MADELON', filename_prefix='ica_madelon')
    em(50, X=ica_credit, dataset='Credit', filename_prefix='ica_credit')
    
    # PCA experiments
    pca_madelon = pca(5, X_madelon)
    pca_credit =  pca(5, X_credit)
    kmeans(50, X=pca_madelon, y=y_madelon, dataset='MADELON', filename_prefix="pca_madelon")
    kmeans(50, X=pca_credit, y=y_credit, dataset='Credit', filename_prefix="pca_credit")
    em(50, X=pca_madelon, dataset='MADELON', filename_prefix='pca_madelon')
    em(50, X=pca_credit, dataset='Credit', filename_prefix='pca_credit')

    # GRP experiments
    grp_madelon = grp(5, X_madelon)
    grp_credit =  grp(5, X_credit)
    kmeans(50, X=grp_madelon, y=y_madelon, dataset='MADELON', filename_prefix="grp_madelon")
    kmeans(50, X=grp_credit, y=y_credit, dataset='Credit', filename_prefix="grp_credit")
    em(50, X=grp_madelon, dataset='MADELON', filename_prefix='grp_madelon')
    em(50, X=grp_credit, dataset='Credit', filename_prefix='grp_credit')

    # KPCA experiments
    kpca_madelon = kpca(X_madelon)
    kpca_credit =  kpca(X_credit)
    kmeans(50, X=kpca_madelon, y=y_madelon, dataset='MADELON', filename_prefix="kpca_madelon")
    kmeans(50, X=kpca_credit, y=y_credit, dataset='Credit', filename_prefix="kpca_credit")
    em(50, X=kpca_madelon, dataset='MADELON', filename_prefix='kpca_madelon')
    em(50, X=kpca_credit, dataset='Credit', filename_prefix='kpca_credit')

    run_dnn_experiments()

if __name__ == '__main__':
    # run_experiments()
    run_dnn_experiments(plots=False)