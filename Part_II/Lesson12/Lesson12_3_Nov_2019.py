#Lesson 12
#
# Page : 92/179
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson12") # pylint: disable=anomalous-backslash-in-string

def load_data():
    '''Loads the Pima Indians Dataset'''
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    x = array[:,0:8]
    y = array[:,8]
    return (x,y)

def prepare_models():
    '''Prepare the models and store them in a list'''
    models = []
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    models.append(('LR',LogisticRegression()))
    models.append(('LDA',LinearDiscriminantAnalysis()))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('CART',DecisionTreeClassifier()))
    models.append(('NB',GaussianNB()))
    models.append(('SVM',SVC()))
    return models

def model_evaluation():
    '''Evaluate each model in turn'''
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    models = prepare_models()
    x,y = load_data()
    results = []
    names = []
    msg = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = KFold(n_splits=10,random_state=7)
        cv_results = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg.append("{}: mean = {}, std = {}".format(name,cv_results.mean(), cv_results.std()))
    return results,names,msg

def plot_results():
    from matplotlib import pyplot
    results,names,msg = model_evaluation()
    print(msg)
    fig = pyplot.figure()
    fig.suptitle("Algorithm Comparision")
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.savefig('Algorithm Comparision.png')
    return "Done saving figure"