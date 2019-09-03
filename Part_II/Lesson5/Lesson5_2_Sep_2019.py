#Lesson 4
#
# Page : 40/179
#
import os
from matplotlib import pyplot as pl
from pandas import read_csv
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson5")
def load_data():
    """Loads the csv file data of the pima indians diabetes dataset"""
    filename = 'pima-indians-diabetes.data.csv'
    names = ['preg','plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names = names)
    return data

def histogram():
    """Create a histogram of the data distribution of each variable"""
    data = load_data()
    data.hist()
    pl.show()
    return "Histogram is plotted"

def density_plots():
    """Create denisty plots of each variable in the distribution"""
    data = load_data()
    data.plot(kind = 'density', subplots = True, layout = (3,3), sharex = False)
    pl.show()
    return "Density Plot is plotted"

def box_plots():
    """Create boxplots of each variable in the distribution"""
    data = load_data()
    data.plot(kind = 'box', subplots = True, layout = (3,3), sharex = False, sharey = False)
    pl.show()
    return "Box Plot is plotted"
def correlation_matrix_plot():
    """Create a correlation matrix of all the variables"""
    import numpy
    data = load_data()
    correlations = data.corr()
    #plot correlation matrix
    fig = pylot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin = -1, vmax = 1)
    fi.colorbar(cax)
    ticks = numpy.arange(0,9,1)
    