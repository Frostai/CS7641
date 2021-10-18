import pandas as pd
import numpy as np
import mlrose_hiive as mlrose
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from plotting import plot_nn_curve_df, plot_learning_curve

VERBOSE=True
SEED = 42

poker_train_df = pd.read_csv('../datasets/poker/poker-hand-training-true.data', sep=',')
poker_test_df = pd.read_csv('../datasets/poker/poker-hand-testing.data', sep=',')

poker_train_df = shuffle(poker_train_df)

X_train = poker_train_df.values[:, :-1]
y_train = poker_train_df.values[:,-1]

X_test = poker_test_df.values[:, :-1]
y_test = poker_test_df.values[:,-1]

# One hot encode target values
encoder = OneHotEncoder()
y_train_hot = encoder.fit_transform(y_train.reshape(-1, 1)).todense()
y_test_hot = encoder.transform(y_test.reshape(-1, 1)).todense()

dnn1 = mlrose.NeuralNetwork(hidden_nodes = [100,100], activation = 'relu', \
                                 algorithm = 'random_hill_climb', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 1, \
                                 early_stopping = True, max_attempts = 100, \
                                 random_state = SEED, curve=True)

dnn2 = mlrose.NeuralNetwork(hidden_nodes = [100,100], activation = 'relu', \
                                 algorithm = 'simulated_annealing', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 1, \
                                 early_stopping = True, max_attempts = 100, \
                                 random_state = SEED, curve=True)

dnn3 = mlrose.NeuralNetwork(hidden_nodes = [100,100], activation = 'relu', \
                                 algorithm = 'genetic_alg', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 1, \
                                 early_stopping = True, max_attempts = 100, \
                                 mutation_prob=0.2, random_state = SEED, curve=True)

def dnnfit1():
    print('RHC Start...')
    # results1 = dnn1.fit(X_train, y_train_hot)
    plots = plot_learning_curve(dnn1,"RHC", X_train, y_train_hot)
    plots.savefig('rhc_dnn.png')
    
    # print(results1.fitness_curve)
    # plot_nn_curve_df(results1.fitness_curve)
    print('RHC')
    # print('Train data Score: ', dnn1.score(X_train, y_train_hot))
    # print('Test data Score', dnn1.score(X_test, y_test_hot))

def dnnfit2():
    print('SA START...')
    plots = plot_learning_curve(dnn2,"SA", X_train, y_train_hot)
    plots.savefig('sa_dnn.png')

def dnnfit3():
    print('GA START...')
    plots = plot_learning_curve(dnn2,"GA", X_train, y_train_hot)
    plots.savefig('ga_dnn.png')
    # dnn3.fit(X_train, y_train_hot)
    # print('GA')
    # print('Train data Score: ', dnn3.score(X_train, y_train_hot))
    # print('Test data Score', dnn3.score(X_test, y_test_hot))

# dnnfit1()
dnnfit2()
dnnfit3()