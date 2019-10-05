#Lesson 8
#
# Page : 66/179
#
import os
from pandas import read_csv
from numpy import set_printoptions
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson8")

def load_data():
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    return array

def split_and_train():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    test_size = 0.33
    seed = 7
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = test_size, random_state = seed)
    model = LogisticRegression()
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    accuracy = result*100.0
    return accuracy

def k_fold_cross_validation():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    num_folds = 10
    seed = 7
    k_fold = KFold(n_splits = num_folds, random_state = seed)
    model = LogisticRegression()
    result = cross_val_score(model, x, y, cv = k_fold)
    accuracy, standard_deviation = result.mean() * 100.0, result.std() * 100.0
    return accuracy, standard_deviation

def leave_one_out_cross_validation():
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    array = load_data()
    x = array[:, 0:8]
    y = array[:,8]
    #num_folds = 10 <----Unnecessary variable
    loocv = LeaveOneOut()
    model = LogisticRegression()
    result = cross_val_score(model, x, y, cv = loocv)
    accuracy, standard_deviation = result.mean() * 100.0, result.std() * 100.0
    return accuracy, standard_deviation

def repeated_random_test_train_splits():
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    array = load_data()
    x = array[:, 0:8]
    y = array[:,8]
    n_splits = 10
    test_size = 0.33
    seed = 7
    kfold = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = seed )
    model = LogisticRegression()
    result = cross_val_score(model, x, y, cv = kfold)
    accuracy, standard_deviation = result.mean() * 100.0, result.std() * 100.0
    return accuracy, standard_deviation