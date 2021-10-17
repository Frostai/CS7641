import pandas as pd
from matplotlib import pyplot as plt

# TEST_PATH = './six_peaks/GA_final/ga__GA_final__curves_df.csv'
def plot_ro_curve(filepath, title):
    # data = pd.read_csv(TEST_PATH, sep=',')
    data = pd.read_csv(filepath=filepath, sep=',')
    print (data)
    plt.plot(data["Iteration"], data["Fitness"])
    plt.title(title)
    plt.show()

def plot_ro_curve_df(data, title, folder):
    plt.plot(data["Iteration"], data["Fitness"])
    plt.title(title)
    plt.savefig(folder+'/'+folder+'_'+title+'.png')
    plt.clf()

def plot_nn_curve_df(data, title=""):
    plt.plot(data)
    plt.title(title)
    plt.show()
