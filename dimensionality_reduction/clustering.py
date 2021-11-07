import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score

VERBOSE=False

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

# KMEANS
def kmeans(K, X, dataset = '', filename_prefix=''):
    print("Start K-means", end='')
    scores = {
        'silhouette': [],
        'davies_bouldin': []
    }
    RANGE=range(2,K)
    for k in RANGE:
        print(".", end='', flush=True)
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=10000, verbose=VERBOSE)
        kmeans.fit(X)
        scores['silhouette'].append(silhouette_score(X, kmeans.labels_))
        scores['davies_bouldin'].append(davies_bouldin_score(X, kmeans.labels_))
    print()
    # Plots
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    
    axes[0].set_title('Silhouette Score for K-means with {} dataset'.format(dataset))
    axes[0].plot(RANGE, scores['silhouette'], label='silhouette')
    axes[0].set_xlabel('K')
    axes[0].set_ylabel('Score')

    axes[1].set_title('Davies Bouldin Score for K-means with {} dataset'.format(dataset))
    axes[1].plot(RANGE, scores['davies_bouldin'], label='davies_bouldin')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('Score')
    plt.savefig('figures/{}_kmeans.png'.format(filename_prefix))
    plt.clf()


# EM
def em(N, X, dataset = '', filename_prefix=''):
    print("Start EM", end='')
    em_scores = {
        'aic': [],
        'bic': []
    }
    N_ARR = range(2,N)
    for n in N_ARR:
        print(".", end='', flush=True)
        gm = GaussianMixture(n_components=n, n_init=10,  covariance_type='full', )
        gm.fit(X)
        em_scores['aic'].append(gm.aic(X))
        em_scores['bic'].append(gm.bic(X))
    print()
    # Plots
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].set_title('GMM AIC on {} dataset'.format(dataset))
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('AIC')
    axes[0].plot(N_ARR, em_scores['aic'], label='aic')

    axes[1].set_title('GMM BIC on {} dataset'.format(dataset))
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('BIC')
    axes[1].plot(N_ARR, em_scores['bic'], label='bic')
    plt.savefig('figures/{}_em.png'.format(filename_prefix))
    plt.clf()


if __name__ == '__main__':
    kmeans(50, X=X_madelon, dataset='MADELON', filename_prefix="base_madelon")
    em(50, X=X_madelon, dataset='MADELON', filename_prefix='base_madelon')
    kmeans(50, X=X_credit, dataset='Credit', filename_prefix="base_credit")
    em(50, X=X_credit, dataset='Credit', filename_prefix='base_credit')