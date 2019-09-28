#Lesson 7
#
# Page : 61/179
#
import os
from pandas import read_csv
from numpy import set_printoptions

os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson6")

def load_data():
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names)
    array = dataframe.values
    return array

def select_k_best():
    '''feature selectiion with Univariate satistical tests chi-squared for classification
    returns a 2d array of 2 arrays: scores_ and features'''
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    #load data
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    #feature selection
    test = SelectKBest(score_func=chi2, k = 4)
    fit = test.fit(x,y)
    #summarize scores
    return_this = []
    return_this.append(fit.scores_)
    features = fit.transform(x)
    return_this.append(features[0:5,:])
    return return_this

def recursive_feature_elimination():
    '''feature selection with recursive feature elimination
    returns 3 strings: number of features, selected features, and ranking'''
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    model = LogisticRegression()
    rfe = RFE(model,3)
    fit = rfe.fit(x,y)
    string = "Num features : {}".format(fit.n_features_)
    return_this = []
    return_this.append(string)
    string = "Selected Features: {}".format(fit.support_)
    return_this.append(string)
    string = "Feature Ranking: {}".format(fit.ranking_)
    return_this.append(string)
    return return_this

def principal_component_analysis():
    '''Feature decomposition(Reduction) via PCA
    returns 2 strings about explained variance and components'''
    from sklearn.decomposition import PCA
    #load data
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    pca = PCA(n_components = 3)
    fit = pca.fit(x)
    string = "Explained Variance: {}".format(fit.explained_variance_ratio_)
    return_this = []
    return_this.append(string)
    string = "Components: {}".format(fit.components_)
    return_this.append(string)
    return return_this

def extra_trees():
    '''Feature importance estimation vai the extra Trees classifier
    returns a list of feature importances'''
    #load data
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    from sklearn.ensemble import ExtraTreesClassifier
    model = ExtraTreesClassifier()
    model.fit(x,y)
    return model.feature_importances_