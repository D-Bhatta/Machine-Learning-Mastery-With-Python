#Lesson 15
#
# Page : 107/179 
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson15") # pylint: disable=anomalous-backslash-in-string

def load_data():
    '''Loads the Pima Indians Dataset'''
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    x = array[:,0:8]
    y = array[:,8]
    return (x,y)

def grid_search():
    x,y = load_data()
    from numpy import array
    alphas = array([1,0.1,0.01,0.001,0.0001,0])
    param_grid = dict(alpha=alphas)
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import Ridge
    model = Ridge()
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid.fit(x,y)
    return grid.best_score_, grid.best_estimator_.alpha

def random_search():
    x,y = load_data()
    from scipy.stats import uniform
    params_grid = {'alpha' : uniform()}
    from sklearn.linear_model import Ridge
    model = Ridge()
    from sklearn.model_selection import RandomizedSearchCV
    rndsearch = RandomizedSearchCV(estimator=model, param_distributions=params_grid, n_iter=100, random_state=7)
    rndsearch.fit(x,y)
    return rndsearch.best_score_, rndsearch.best_estimator_.alpha


