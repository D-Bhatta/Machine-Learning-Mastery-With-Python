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

from  Lesson11 import Lesson11_27_Oct_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data_housing(self):
        array = str(ls.load_data_housing())
        array_test = temp.temp_load_data_housing()
        assert array == array_test, "Couldn't load Housing dataset"
    
    def test_spot_check_linear_linear_regression(self):
        mean = ls.spot_check_linear_linear_regression()
        assert mean == -34.70525594452509, "Linear regression spot check values do not match at spot_check_linear_linear_Regression()"

    def test_spot_check_linear_ridge_regression(self):
        mean = ls.spot_check_linear_ridge_regression()
        assert mean == -34.07824620925929, "Ridge regression spot check values do not match at spot_check_linear_ridge_regression()"

    def test_spot_check_linear_lasso_regression(self):
        mean = ls.spot_check_linear_lasso_regression()
        assert mean == -34.46408458830232, "Linear regression spot check values do not match at spot_check_linear_lasso_regression()"

    def test_spot_check_linear_elastic_net_regression(self):
        mean = ls.spot_check_linear_elastic_net_regression()
        assert mean == -31.164573714249777, "ElastiCNet regression spot check values do not match at spot_check_linear_elastic_net_regression()"

    def test_spot_check_non_linear_knn_regression(self):
        mean = ls.spot_check_non_linear_knn_regression()
        assert mean == -107.28683898039215, "KNN regression spot check values do not match at spot_check_non_linear_knn_regression()"

    def test_spot_check_non_linear_cart_regression(self):
        mean = ls.spot_check_non_linear_cart_regression()
        assert mean > -45 and mean < -30, "CART regression spot check values do not match at spot_check_non_linear_cart_regression()"

    def test_spot_check_non_linear_svm_regression(self):
        mean = ls.spot_check_non_linear_svm_regression()
        assert mean == -91.04782433324428, "SVM regression spot check values do not match at spot_check_non_linear_svm_regression()"