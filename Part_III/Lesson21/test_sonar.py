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
    
