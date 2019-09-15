#Lesson 6
#
# Page : 56/179
#
import os
from pandas import read_csv
from numpy import set_printoptions

os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson6")

def load_data():
    filename = "pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin', 'test' , 'mass', 'pedi', 'age', 'class']
    dataframe = read_csv(filename, names = names)
    array = dataframe.values
    return array

def rescale_data():
    from sklearn.preprocessing import MinMaxScaler
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledx = scaler.fit_transform(x)
    return rescaledx,rescaledx[0:5,:]

def strandardize_data():
    from sklearn.preprocessing import StandardScaler
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    scaler = StandardScaler().fit(x)
    rescaledx = scaler.transform(x)
    return rescaledx, rescaledx[0:5,:]

def normalize_data():
    from sklearn.preprocessing import Normalizer
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    scaler = Normalizer().fit(x)
    normalizedx = scaler.transform(x)
    return normalizedx, normalizedx[0:5,:]

def binarize_data():
    from sklearn.preprocessing import Binarizer
    array = load_data()
    x = array[:,0:8]
    y = array[:,8]
    binarizer = Binarizer(threshold = 0.0).fit(x)
    binaryx = binarizer.transform(x)
    return binaryx, binaryx[0:5,:]