#testcases for Lesson8_4_Oct_2019.py
import numpy
import os
import io
import sys
import temp
import pytest
import numpy as np
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson10 import Lesson10_27_Oct_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data_pima_indians(self):
        array = str(ls.load_data_pima_indians())
        array_test = temp.temp_dataframe_values()
        assert array == array_test, "Couldn't load Pima Indians Dataset"
    
    def test_spot_check_linear_Logistic_Regression(self):
        mean = ls.spot_check_linear_Logistic_Regression()
        assert mean == 0.7695146958304853, "Logistic regression spot check values do not match at spot_check_linear_Logistic_Regression()"

    def test_spot_check_linear_LDA(self):
        mean = ls.spot_check_linear_LDA()
        assert mean == 0.773462064251538, "LDA spot check values do not match at spot_check_linear_LDA()"

    def test_spot_check_non_linear_knn(self):
        mean = ls.spot_check_non_linear_knn()
        assert mean == 0.7265550239234451, "Knn spot check values do not match at spot_check_non_linear_knn()"

    def test_spot_check_non_linear_naive_bayes(self):
        mean = ls.spot_check_non_linear_naive_bayes()
        assert mean == 0.7551777170198223, "Naive Bayes spot check values do not match at spot_check_non_linear_naive_bayes()"

    def test_spot_check_non_linear_cart(self):
        mean = ls.spot_check_non_linear_cart()
        assert mean > 0.67, "CART spot check values do not match at spot_check_non_linear_cart()"

    def test_spot_check_non_linear_svm(self):
        mean = ls.spot_check_non_linear_svm()
        assert mean == 0.6510252904989747, "SVM spot check values do not match at spot_check_non_linear_svm()"