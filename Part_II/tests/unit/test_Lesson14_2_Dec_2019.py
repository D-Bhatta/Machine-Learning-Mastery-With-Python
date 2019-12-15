#testcases for Lesson14_2_Dec_2019.py
import sys
import temp
from numpy import array
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson14 import Lesson14_2_Dec_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data(self):
        x,y = ls.load_data()
        x = str(x)
        y = str(y)
        temp_x,temp_y = temp.temp_load_x_pima(),temp.temp_load_y_pima()
        assert (x,y == temp_x,temp_y), "Couldn't load Pima data x and y values at load_data()"
    def test_bagging(self):
        results = ls.bagging()
        assert results == 0.770745044429255, "The mean of the cross validation score results doesn't match anticipated value at bagging()"

    def test_rand_forest(self):
        results = ls.rand_forest()
        assert results >= 0.76, "The mean of the cross validation score results doesn't match anticipated value at rand_forest()"

    def test_extra_trees(self):
        results = ls.extra_trees()
        assert results >= 0.75, "The mean of the cross validation score results doesn't match anticipated value at extra_trees()"

    def test_adaBoost(self):
        results = ls.adaBoost()
        assert results == 0.760457963089542, "The mean of the cross validation score results doesn't match anticipated value at adaBoost()"

    def test_sto_grad_boost(self):
        results = ls.sto_grad_boost()
        assert results >= 0.75, "The mean of the cross validation score results doesn't match anticipated value at sto_grad_boost()"

    def test_voting_ensemble(self):
        results = ls.voting_ensemble()
        assert results >= 0.72, "The mean of the cross validation score results doesn't match anticipated value at voting_ensemble()"