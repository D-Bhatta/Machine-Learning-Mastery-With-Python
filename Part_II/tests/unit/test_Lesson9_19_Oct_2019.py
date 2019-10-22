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

from  Lesson9 import Lesson9_19_Oct_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data_pima_indians(self):
        array = str(ls.load_data_pima_indians())
        array_test = temp.temp_dataframe_values()
        assert array == array_test, "Couldn't load Pima Indians Dataset"

    def test_classification_accuracy(self):
        mean, std = ls.classification_accuracy()
        assert (mean, std) == (76.95146958304852, 4.841051924567195), "Classification accuracy values do not match at classification_accuracy()"

    def test_logarithmic_loss(self):
        mean, std = ls.logarithmic_loss()
        assert (mean, std) == (-0.4925017098423125, 0.04703579370752684), "Logarithmic Loss values do not match at logarithmic_loss()"

    def test_area_under_roc_curve(self):
        mean, std = ls.area_under_roc_curve()
        assert (mean, std) == (0.823716379293716, 0.040723558409611726), "Area under ROC Curve values do not match at area_under_roc_curve()"

    def test_confusion_matrix(self):
        matrix = ls.confusion_matrix()
        matrix = matrix.tolist()
        assert matrix == [[141, 21], [41, 51]], "Confusion matrix values do not match at confusion_matrix()"

    def test_classification_report(self):
        report = ls.classification_report()
        assert report == '              precision    recall  f1-score   support\n\n         0.0       0.77      0.87      0.82       162\n         1.0       0.71      0.55      0.62        92\n\n    accuracy                           0.76       254\n   macro avg       0.74      0.71      0.72       254\nweighted avg       0.75      0.76      0.75       254\n',"Classification report string doesn't match"

    def test_load_data_housing(self):
        array = str(ls.load_data_housing())
        array_test = temp.temp_load_data_housing()
        assert array == array_test, "Couldn't load Housing dataset"

    def test_mean_absolute_error(self):
        mean, std = ls.mean_absolute_error()
        assert (mean, std) == (-4.004946635324019, 2.0835992687095204), "Mean Absolute Error values do not match at mean_absolute_error()"

    def test_mean_squared_error(self):
        mean, std = ls.mean_squared_error()
        assert (mean, std) == (-34.70525594452509, 45.57399920030867), "Mean Squared Error values do not match at mean_squared_error()"

    def test_r_squared(self):
        mean, std = ls.r_squared()
        assert (mean, std) == (0.20252899006054745, 0.5952960169512453), "R squared Error values do not match at r_squared()"