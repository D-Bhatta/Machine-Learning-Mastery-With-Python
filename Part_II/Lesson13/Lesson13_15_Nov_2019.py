#Lesson 13
#
# Page : 96/179
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
from sklearn.pipeline import Pipeline
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson13") # pylint: disable=anomalous-backslash-in-string

def load_data():
    '''Loads the Pima Indians Dataset'''
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    x = array[:,0:8]
    y = array[:,8]
    return (x,y)

def create_pipeline_data_preparation():
    '''Creates a pipeline for the data preparation stage'''
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('lda', LinearDiscriminantAnalysis()))
    model = Pipeline(estimators)
    return model

def evaluate_pipeline_data_preparation():
    x,y = load_data()
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    model = create_pipeline_data_preparation()
    kfold = KFold(n_splits=10,random_state=10)
    results = cross_val_score(model,x,y,cv=kfold)
    return results.mean()

def create_pipeline_feature_extraction():
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.linear_model import LogisticRegression
    features = []
    features.append(('pca', PCA(n_components=3)))
    features.append(('select_best', SelectKBest(k=6)))
    feature_union = FeatureUnion(features)
    estimators = []
    estimators.append(('feature_union', feature_union))
    estimators.append(('logistic', LogisticRegression()))
    model = Pipeline(estimators)
    return model

def evaluate_pipeline_feature_extraction():
    x,y = load_data()
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    model = create_pipeline_feature_extraction()
    kfold = KFold(n_splits=10, random_state=7)
    results = cross_val_score(model,x,y,cv=kfold)
    return results.mean()