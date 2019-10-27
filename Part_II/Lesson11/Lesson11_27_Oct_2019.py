#Lesson 11
#
# Page : 85/179
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
from numpy import set_printoptions
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson11") # pylint: disable=anomalous-backslash-in-string

def load_data_housing():
    from pandas import read_csv
    filename = 'housing.csv'
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B','LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    array = dataframe.values
    return array

def spot_check_linear_linear_regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = LinearRegression()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    return results.mean()

def spot_check_linear_ridge_regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Ridge
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = Ridge()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    return results.mean()

def spot_check_linear_lasso_regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import Lasso
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = Lasso()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    return results.mean()

def spot_check_linear_elastic_net_regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import ElasticNet
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = ElasticNet()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    return results.mean()

def spot_check_non_linear_knn_regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsRegressor
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = KNeighborsRegressor()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    return results.mean()

def spot_check_non_linear_cart_regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = DecisionTreeRegressor()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    return results.mean()

def spot_check_non_linear_svm_regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVR
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeRegressor
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = SVR()
    scoring = 'neg_mean_squared_error'
    results = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
    return results.mean()