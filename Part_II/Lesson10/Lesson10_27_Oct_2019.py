#Lesson 10
#
# Page : 79/179
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
from numpy import set_printoptions
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson10") # pylint: disable=anomalous-backslash-in-string

def load_data_pima_indians():
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    return array

def spot_check_linear_Logistic_Regression():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state = 7)
    model = LogisticRegression()
    results = cross_val_score(model, x,y,cv = kfold)
    return results.mean()

def spot_check_linear_LDA():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state = 7)
    model = LinearDiscriminantAnalysis()
    results = cross_val_score(model, x,y,cv = kfold)
    return results.mean()

def spot_check_non_linear_knn():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state = 7)
    model = KNeighborsClassifier()
    results = cross_val_score(model, x,y,cv = kfold)
    return results.mean()

def spot_check_non_linear_naive_bayes():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state = 7)
    model = GaussianNB()
    results = cross_val_score(model, x,y,cv = kfold)
    return results.mean()

def spot_check_non_linear_cart():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state = 7)
    model = DecisionTreeClassifier()
    results = cross_val_score(model, x,y,cv = kfold)
    return results.mean()

def spot_check_non_linear_svm():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.svm import SVC
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state = 7)
    model = SVC()
    results = cross_val_score(model, x,y,cv = kfold)
    return results.mean()