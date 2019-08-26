#testcases for Lesson2_25_Aug_2019.py
import numpy
import sys,os
import io
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson2 import Lesson2_25_Aug_2019 as ls

class TestObject(object):
    def test_numpy_arithmatic_example(self):
        list1 = [[2,2,2],[-2,-2,-2],[5,5,5]]
        list2 = [[3,3,3],[-3,-3,-3],[10,10,10]]
        list3 = ['(array([5, 5, 5]), array([6, 6, 6]))','(array([-5, -5, -5]), array([6, 6, 6]))','(array([15, 15, 15]), array([50, 50, 50]))']
        list4 = []
        for i in range(3):
            a =  ls.numpy_arithmatic_example(list1[i],list2[i])
            list4.append(a)
        for i in range(3):
            assert str(list3[i]) == str(list4[i])

    def testpandas_series_example(self):
        a = ls.pandas_series_example()
        a = str(a)
        assert a == '''a    1\nb    2\nc    3\ndtype: int32'''

    def test_pandas_series_data_access_example(self):
        myseries = ls.pandas_series_example()
        assert ls.pandas_series_data_access_example(myseries) == (1,1)

    def test_pandas_dataframe_example(self):
        a = ls.pandas_dataframe_example()
        a = str(a)
        assert a == '''   one  two  three\na    1    2      3\nb    4    5      6'''

    def test_pandas_dataframe_data_access_example(self):
        mydataframe = ls.pandas_dataframe_example()
        a = ls.pandas_dataframe_data_access_example(mydataframe)
        a = str(a)
        assert a == '''(a    1\nb    4\nName: one, dtype: int32, a    1\nb    4\nName: one, dtype: int32)'''
