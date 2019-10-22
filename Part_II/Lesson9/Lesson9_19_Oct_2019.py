#Lesson 9
#
# Page : 66/179
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
from numpy import set_printoptions
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson9")

def load_data_pima_indians():
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    return array

def classification_accuracy():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state=7)
    model = LogisticRegression()
    scoring = 'accuracy'
    results = cross_val_score(model,x,y,cv = kfold, scoring = scoring, )
    return results.mean()*100, results.std()*100

def logarithmic_loss():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state=7)
    model = LogisticRegression()
    scoring = 'neg_log_loss'
    results = cross_val_score(model,x,y,cv = kfold, scoring = scoring, )
    return results.mean(), results.std()

def area_under_roc_curve():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    kfold = KFold(n_splits = 10, random_state=7)
    model = LogisticRegression()
    scoring = 'roc_auc'
    results = cross_val_score(model,x,y,cv = kfold, scoring = scoring, )
    return results.mean(), results.std()

def confusion_matrix():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    test_size = 0.33
    seed = 7
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size, random_state = seed)
    model = LogisticRegression()
    model.fit(x_train,y_train)
    predicted = model.predict(x_test)
    matrix = confusion_matrix(y_test, predicted)
    return matrix

def classification_report():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    array = load_data_pima_indians()
    x = array[:,0:8]
    y = array[:,8]
    test_size = 0.33
    seed = 7
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size, random_state = seed)
    model = LogisticRegression()
    model.fit(x_train,y_train)
    predicted = model.predict(x_test)
    report = classification_report(y_test, predicted)
    return report

def load_data_housing():
    from pandas import read_csv
    filename = 'housing.csv'
    names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B','LSTAT', 'MEDV']
    dataframe = read_csv(filename, delim_whitespace=True, names=names)
    array = dataframe.values
    return array

def mean_absolute_error():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = LinearRegression()
    scoring  = 'neg_mean_absolute_error'
    results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    return results.mean(), results.std()

def mean_squared_error():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = LinearRegression()
    scoring  = 'neg_mean_squared_error'
    results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    return results.mean(), results.std()

def r_squared():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LinearRegression
    array = load_data_housing()
    x = array[:,0:13]
    y = array[:,13]
    kfold = KFold(n_splits=10, random_state=7)
    model = LinearRegression()
    scoring  = 'r2'
    results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
    return results.mean(), results.std()