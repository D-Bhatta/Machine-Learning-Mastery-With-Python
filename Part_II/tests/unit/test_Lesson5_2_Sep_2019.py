#testcases for Lesson3_26_Aug_2019.py
import numpy
import sys,os
import io
import sys
import temp
import pytest
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson5 import Lesson5_2_Sep_2019 as ls

class TestObject(object):
    def test_load_data(self):
        """Loads the csv file data of the pima indians diabetes dataset"""
        x = ls.load_data()
        y = temp.temp_pandas()
        x = str(x)
        assert x == y

    def test_histogram(self):
        """Create a histogram of the data distribution of each variable"""
        a = ls.histogram()
        assert a == "Histogram is plotted"
        

    def test_density_plots(self):
        """Create denisty plots of each variable in the distribution"""
        a = ls.density_plots()
        assert a == "Density Plot is plotted"
        

    def test_box_plots(self):
        """Create boxplots of each variable in the distribution"""
        a = ls.box_plots()
        assert a == "Box Plot is plotted"

    def test_correlation_matrix_plot(self):
        """Create a correlation matrix of all the variables"""
        a = ls.correlation_matrix_plot()
        assert a == "Correlation Matrix has been plotted"

    def test_scatter_plot(self):
        """Create a Scatter Plot Matrix of all the variables"""
        a = ls.scatter_plot()
        assert a == "Scatter Plot matrix has been plotted"