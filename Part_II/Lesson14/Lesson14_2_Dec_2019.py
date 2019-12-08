#Lesson 14
#
# Page : 100/179
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
from sklearn.pipeline import Pipeline
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson14") # pylint: disable=anomalous-backslash-in-string

def load_data():
    '''Loads the Pima Indians Dataset'''
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    x = array[:,0:8]
    y = array[:,8]
    return (x,y)

def bagging():
    x,y = load_data()
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier
    seed = 7
    kfold = KFold(n_splits=10, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees,random_state=seed)
    results = cross_val_score(model, x, y, cv=kfold)

    return results.mean()

def rand_forest():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    x,y = load_data()
    num_trees = 100
    seed = 7
    max_features = 3
    kfold = KFold(n_splits=10, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = cross_val_score(model, x, y, cv=kfold)
    return results.mean()

def extra_trees():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    x,y = load_data()
    num_trees = 100
    max_features = 7
    kfold = KFold(n_splits=10, random_state=7)
    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
    results = cross_val_score(model,x,y,cv=kfold)
    return results.mean()

def adaBoost():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import AdaBoostClassifier
    x,y = load_data()
    num_trees = 30
    seed = 7
    kfold = KFold(n_splits=10, random_state=7)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = cross_val_score(model,x,y,cv=kfold)
    return results.mean()

def sto_grad_boost():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingClassifier
    x,y = load_data()
    seed = 7
    num_trees = 100
    kfold = KFold(n_splits=10, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=num_trees,)
    results = cross_val_score(model,x,y,cv=kfold)
    return results.mean()

def voting_ensemble():
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble.voting import VotingClassifier
    x,y = load_data()
    kfold = KFold(n_splits=10, random_state=7)
    #create the sub models
    estimators = []
    model_1 = LogisticRegression()
    model_2 = DecisionTreeClassifier()
    model_3 = SVC()
    estimators.append(('logistic',model_1))
    estimators.append(('cart', model_2))
    estimators.append(('svm', model_3))
    #create the ensemble model
    ensemble = VotingClassifier(estimators)
    results = cross_val_score(ensemble,x,y,cv=kfold)
    return results.mean()