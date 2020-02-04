import pandas
import numpy

import boston
B = boston.Boston()

class TestObject(object):
    def test_dataset_loading(self):
        assert B.dataset.size != 0, " TEST FAILED: Failed to load dataset"