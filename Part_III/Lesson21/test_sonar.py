import sonar
from pandas import read_csv
sn = sonar.Sonar()
class TestObject(object):
    def test_test(self):
        assert 1 == 1, "Something is wrong with the test"
    
    def test_data_loading(self):
        data = sn.data
        url = 'sonar/sonar.all-data.csv'
        test_data = read_csv(url, header=None)
        assert str(test_data) == str(data), "Data didn't load correctly"
    
    def test_validation(self):
        x = sn.x
        assert x.size > 0 , "self.x is empty"
        x_t = sn.x_train
        assert x_t.size > 0 , "self.x_train is empty"
        x_v = sn.x_validation
        assert x_v.size > 0 , "self.x_validation is empty"
        y = sn.y
        assert y.size > 0 , "self.y is empty"
        y_t = sn.y_train
        assert y_t.size > 0 , "self.y_train is empty"
        y_v = sn.y_validation
        assert y_v.size > 0 , "self.y_validation is empty"
        

        
