#testcases for Lesson16_15_Dec_2019.py
import sys
import temp
from numpy import array
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson16 import Lesson16_15_Dec_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data(self):
        x,y = ls.load_data()
        x = str(x)
        y = str(y)
        temp_x,temp_y = temp.temp_load_x_pima(),temp.temp_load_y_pima()
        assert (x,y == temp_x,temp_y), "Couldn't load Pima data x and y values at load_data()"
    
    def test_finalize_pickle(self):
        result = ls.finalize_pickle()
        assert result == 0.7559055118110236, "The result value doesn't match the anticipated value at finalize_pickle()"
    def test_finalize_joblib(self):
        result = ls.finalize_joblib()
        assert result ==  0.7559055118110236, "The result value doesn't match the anticipated value at finalize_joblib()"