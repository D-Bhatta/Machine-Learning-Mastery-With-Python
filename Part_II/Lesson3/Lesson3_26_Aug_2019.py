##Lesson 3
#
# Page : 35/179
#
#We will be using the Pima Indians Dataset
#Load the CSV dataset
import csv
import numpy
import os
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson3")

def csv_to_list():
    filename = 'pima-indians-diabetes.data.csv'
    with open(filename) as f:
        reader = csv.reader(f, delimiter = ',', quoting = csv.QUOTE_NONE)
        x = list(reader)
    return x
    
def list_to_Numpy_array():
    x = csv_to_list()
    data = numpy.array(x).astype('float')
    return data

def print_data(data):
    print(data.shape)

#Load CSv Files with NumPy
def load_csv_numpy():
    from numpy import loadtxt
    filename = 'pima-indians-diabetes.data.csv'
    with open(filename, 'rb') as f:
        data = loadtxt(f,delimiter = ",")
    return data
'''
def load_csv_numpy_url():
    from numpy import loadtxt
    from urllib import request
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    with request.urlopen(url) as response:
        html = response.read()
    with open(html, 'rb') as f:
        data = loadtxt(f,delimiter = ",")
    return data'''
#Load CSV files with Pandas

def load_csv_pandas():
    from pandas import read_csv
    filename = 'pima-indians-diabetes.data.csv'
    name_cols = ['preg', 'plas','pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filename, names=name_cols)
    return data
'''
def load_csv_pandas_url_():
    from pandas import read_csv
    from urllib import request
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    name_cols = ['preg', 'plas','pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    with request.urlopen(url) as response:
        html = response.read()
        html = str(html)
        data = read_csv(html, names=name_cols)
    return data'''
