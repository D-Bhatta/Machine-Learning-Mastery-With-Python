#Lesson 15
#
# Page : 110/179 
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson16") # pylint: disable=anomalous-backslash-in-string

def load_data():
    '''Loads the Pima Indians Dataset'''
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    x = array[:,0:8]
    y = array[:,8]
    return (x,y)
    
def finalize_pickle():
    x,y = load_data()
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=7)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_train,y_train)

    filename = 'finalized_pickle_model.sav'

    from pickle import dump

    dump(model, open(filename,'wb'))

    from pickle import load
    loaded_model = load(open(filename, 'rb'))
    result = loaded_model.score(x_test,y_test)
    return result

def finalize_joblib():
    x,y = load_data()
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=7)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(x_train,y_train)
    filename = 'finalized_joblib_model.sav'
    from joblib import dump
    dump(model, filename)
    from joblib import load
    loaded_model = load(filename)

    result = loaded_model.score(x_test,y_test)
    return result