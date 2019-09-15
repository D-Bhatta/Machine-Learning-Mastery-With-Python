#testcases for Lesson3_26_Aug_2019.py
import numpy
import os
import io
import sys
import temp
import pytest
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson6 import Lesson6_9_Sep_2019 as ls

class TestObject(object):
    def test_load_data(self):
        array = str(ls.load_data())
        array_test = temp.temp_dataframe_values()
        assert array == array_test

    def test_rescale_data(self):
        a = str(ls.rescale_data())
        assert a == temp.temp_rescaled_data()

    def test_strandardize_data(self):
        a = str(ls.strandardize_data())
        assert a == temp.temp_standardized_data()

    def test_normalize_data(self):
        a = str(ls.normalize_data())
        assert a == temp.temp_normalize_data()

    def test_binarize_data(self):
        a = str(ls.binarize_data())
        assert a == temp.temp_binarize_data()