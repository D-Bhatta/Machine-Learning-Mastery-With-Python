#testcases for Lesson15_15_Dec_2019.py
import sys
import temp
from numpy import array
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson15 import Lesson15_15_Dec_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data(self):
        x,y = ls.load_data()
        x = str(x)
        y = str(y)
        temp_x,temp_y = temp.temp_load_x_pima(),temp.temp_load_y_pima()
        assert (x,y == temp_x,temp_y), "Couldn't load Pima data x and y values at load_data()"
    
    def test_grid_search(self):
        best_score, alpha = ls.grid_search()
        assert (best_score,alpha) == (0.2796175593129723, 1.0), "The best score and alpha results don't match anticipated value at grid_search()"

    def test_random_search(self):
        best_score, alpha = ls.random_search()
        assert (best_score,alpha) == (0.27961712703051095, 0.9779895119966027), "The best score and alpha results don't match anticipated value at random_search()"