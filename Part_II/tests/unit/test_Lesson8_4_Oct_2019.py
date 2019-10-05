#testcases for Lesson8_4_Oct_2019.py
import numpy
import os
import io
import sys
import temp
import pytest
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson8 import Lesson8_4_Oct_2019 as ls

class TestObject(object):
    def test_load_data(self):
        array = str(ls.load_data())
        array_test = temp.temp_dataframe_values()
        assert array == array_test

    def test_split_and_train(self):
        accuracy = ls.split_and_train()
        assert accuracy == 75.59055118110236

    def test_k_fold_cross_validation(self):
        accuracy, standard_deviation = ls.k_fold_cross_validation()
        assert accuracy == 76.95146958304852
        assert standard_deviation == 4.841051924567195

    def test_leave_one_out_cross_validation(self):
        accuracy, standard_deviation = ls.leave_one_out_cross_validation()
        assert accuracy == 76.82291666666666
        assert standard_deviation == 42.1963403803346

    def test_repeated_random_test_train_splits(self):
        accuracy, standard_deviation = ls.repeated_random_test_train_splits()
        assert accuracy == 76.49606299212599
        assert standard_deviation == 1.6983980007970874