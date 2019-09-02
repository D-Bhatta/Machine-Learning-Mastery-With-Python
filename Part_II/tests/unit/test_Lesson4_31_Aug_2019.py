#testcases for Lesson3_26_Aug_2019.py
import numpy
import sys,os
import io
import sys
import temp
import pytest
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson4 import Lesson4_31_Aug_2019 as ls

class TestObject(object):
    def test_load_data(self):
        """Load the csv file as a pandas dataframe"""
        x = ls.load_data()
        y = temp.temp_pandas()
        x = str(x)
        assert x == y

    def test_data_peek(self):
        '''view first 20 rows'''
        x = ls.data_peek()
        y = temp.temp_peek()
        x = str(x)
        assert x == y
        
    def test_data_dimensions(self):
        '''View the data dimensions'''
        x = ls.data_dimensions()
        assert x == (768,9)#dimensions of the data are 768 rows and 9 columns
        
    def test_data_datatypes(self):
        '''View the data types present in the data'''
        x = ls.data_datatypes()#the datatypes present in the data are:
        x = str(x)
        assert x == 'preg       int64\nplas       int64\npres       int64\nskin       int64\ntest       int64\nmass     float64\npedi     float64\nage        int64\nclass      int64\ndtype: object'
        
    def test_data_description(self):
        '''View the statistical descriptio of the data'''
        x = ls.data_description()#descriptive statistics of the data
        x = str(x)
        y = '          preg     plas     pres     skin     test     mass     pedi      age  \\\ncount  768.000  768.000  768.000  768.000  768.000  768.000  768.000  768.000   \nmean     3.845  120.895   69.105   20.536   79.799   31.993    0.472   33.241   \nstd      3.370   31.973   19.356   15.952  115.244    7.884    0.331   11.760   \nmin      0.000    0.000    0.000    0.000    0.000    0.000    0.078   21.000   \n25%      1.000   99.000   62.000    0.000    0.000   27.300    0.244   24.000   \n50%      3.000  117.000   72.000   23.000   30.500   32.000    0.372   29.000   \n75%      6.000  140.250   80.000   32.000  127.250   36.600    0.626   41.000   \nmax     17.000  199.000  122.000   99.000  846.000   67.100    2.420   81.000   \n\n         class  \ncount  768.000  \nmean     0.349  \nstd      0.477  \nmin      0.000  \n25%      0.000  \n50%      0.000  \n75%      1.000  \nmax      1.000  '
        assert x == y
        
    def test_data_class_distribution(self):
        '''View the class distribution of the data'''
        x = ls.data_class_distribution() #which class each row falls into, i.e., the number of class 1 and class 2 observations
        x = dict(x)
        y = {0: 500, 1: 268}
        assert x == y
        
    def test_data_correlations(self):
        '''View the data correlation materix generated by calculating the Pearson's Correlation Coefficient'''
        x = ls.data_correlations()# shows which row is correlated to other rows
        x = str(x)
        y = '        preg   plas   pres   skin   test   mass   pedi    age  class\npreg   1.000  0.129  0.141 -0.082 -0.074  0.018 -0.034  0.544  0.222\nplas   0.129  1.000  0.153  0.057  0.331  0.221  0.137  0.264  0.467\npres   0.141  0.153  1.000  0.207  0.089  0.282  0.041  0.240  0.065\nskin  -0.082  0.057  0.207  1.000  0.437  0.393  0.184 -0.114  0.075\ntest  -0.074  0.331  0.089  0.437  1.000  0.198  0.185 -0.042  0.131\nmass   0.018  0.221  0.282  0.393  0.198  1.000  0.141  0.036  0.293\npedi  -0.034  0.137  0.041  0.184  0.185  0.141  1.000  0.034  0.174\nage    0.544  0.264  0.240 -0.114 -0.042  0.036  0.034  1.000  0.238\nclass  0.222  0.467  0.065  0.075  0.131  0.293  0.174  0.238  1.000'
        assert x == y

    def test_data_skew(self):
        '''View the skew matrix for data variables'''
        x = ls.data_skew()
        x = dict(x)        
        y = {'preg': 0.9016739791518673, 'plas': 0.17375350179188992, 'pres': -1.8436079833551302, 'skin': 0.10937249648187539, 'test': 2.2722508584315686, 'mass': -0.4289815884535583, 'pedi': 1.9199110663072108, 'age': 1.1295967011444792, 'class': 0.635016643444981}
        assert x == y