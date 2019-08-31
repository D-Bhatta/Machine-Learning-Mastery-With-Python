#testcases for Lesson3_26_Aug_2019.py
import numpy
import sys,os
import io
import sys
import temp
import pytest
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson3 import Lesson3_26_Aug_2019 as ls

class TestObject(object):
    def test_csv_to_list(self):
        x = ls.csv_to_list()
        y = temp.temp_x()
        assert x == y
    def test_list_to_Numpy_array(self):
        x = ls.list_to_Numpy_array()
        y = temp.temp_y()
        assert str(x) == str(y)
    def test_print_data(self,capsys):
        ls.print_data(ls.list_to_Numpy_array())
        captured = capsys.readouterr()
        assert captured.out == "(768, 9)\n"
    #load with pandas
    def test_load_csv_numpy(self):
        x = ls.load_csv_numpy()
        y = temp.temp_numpy_1()
        x = str(x)
        assert x == y
    '''def test_load_csv_numpy_url(self):
        x = ls.load_csv_numpy_url()
        y = temp.temp_numpy_1()
        x = str(x)
        assert x == y'''
    #load with pandas
    def test_load_csv_pandas(self):
        x = ls.load_csv_pandas()
        y = temp.temp_pandas()
        x = str(x)
        assert x == y
    '''def test_load_csv_pandas_url(self):
        x = ls.load_csv_pandas_url_()
        y = temp.temp_pandas()
        x = str(x)
        assert x == y'''