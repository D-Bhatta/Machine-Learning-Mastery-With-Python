#testcases for Lesson7_28_Sep_2019.py
import numpy
import os
import io
import sys
import temp
import pytest
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson7 import Lesson7_28_Sep_2019 as ls

class TestObject(object):
    def test_load_data(self):
        array = str(ls.load_data())
        array_test = temp.temp_dataframe_values()
        assert array == array_test
    
    def test_select_k_best(self):
        array = ls.select_k_best()
        assert array[0] == [111.51969063588255,  1411.887040644141,  17.605373215320718,  53.10803983632434,  2175.5652729220137,  127.66934333103538,  5.392681546971434,  181.30368904430023]
        assert array[1] == '[[148.    0.   33.6  50. ]\n [ 85.    0.   26.6  31. ]\n [183.    0.   23.3  32. ]\n [ 89.   94.   28.1  21. ]\n [137.  168.   43.1  33. ]]'

    def test_recursive_feature_elimination(self):
        array = ls.recursive_feature_elimination()
        assert array == ['Num features : 3', 'Selected Features: [ True False False False False  True  True False]', 'Feature Ranking: [1 2 3 5 6 1 1 4]']

    def test_principal_component_analysis(self):
        array = ls.principal_component_analysis()
        assert array == ['Explained Variance: [0.88854663 0.06159078 0.02579012]', 'Components: [[-2.02176587e-03  9.78115765e-02  1.60930503e-02  6.07566861e-02\n   9.93110844e-01  1.40108085e-02  5.37167919e-04 -3.56474430e-03]\n [-2.26488861e-02 -9.72210040e-01 -1.41909330e-01  5.78614699e-02\n   9.46266913e-02 -4.69729766e-02 -8.16804621e-04 -1.40168181e-01]\n [-2.24649003e-02  1.43428710e-01 -9.22467192e-01 -3.07013055e-01\n   2.09773019e-02 -1.32444542e-01 -6.39983017e-04 -1.25454310e-01]]']

    '''def test_extra_trees(self):
        array = ls.extra_trees()
        assert array == [0.1075399288945778, 0.2293377156290201, 0.089835408038604, 0.08605215876639975, 0.06855897325295918, 0.14544254072860713, 0.1179941383128816, 0.15523913637695047]'''
        #the above test doesn't work due to random output each time