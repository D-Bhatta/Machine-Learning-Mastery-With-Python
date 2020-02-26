import pandas
import numpy

import boston
import os
B = boston.Boston()

class TestObject(object):
    def test_dataset_loading(self):
        assert B.dataset.size != 0, " TEST FAILED: Failed to load dataset"
    def test_dataset_statistics(self):
        shape = (506, 14)
        dtypes = 'CRIM       float64\nZN         float64\nINDUS      float64\nCHAS         int64\nNOX        float64\nRM         float64\nAGE        float64\nDIS        float64\nRAD          int64\nTAX        float64\nPTRATIO    float64\nB          float64\nLSTAT      float64\nMEDV       float64\ndtype: object'
        description = [['count', 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0, 506.0], ['mean', 3.6135235573122535, 11.363636363636363, 11.136778656126504, 0.0691699604743083, 0.5546950592885372, 6.284634387351787, 68.57490118577078, 3.795042687747034, 9.549407114624506, 408.2371541501976, 18.455533596837967, 356.67403162055257, 12.653063241106723, 22.532806324110698], ['std', 8.601545105332487, 23.322452994515036, 6.8603529408975845, 0.2539940413404118, 0.11587767566755611, 0.7026171434153237, 28.148861406903638, 2.1057101266276104, 8.707259384239377, 168.53711605495926, 2.164945523714446, 91.29486438415779, 7.141061511348571, 9.19710408737982], ['min', 0.00632, 0.0, 0.46, 0.0, 0.385, 3.561, 2.9, 1.1296, 1.0, 187.0, 12.6, 0.32, 1.73, 5.0], ['25%', 0.08204499999999999, 0.0, 5.19, 0.0, 0.449, 5.8855, 45.025, 2.100175, 4.0, 279.0, 17.4, 375.3775, 6.949999999999999, 17.025], ['50%', 0.25651, 0.0, 9.69, 0.0, 0.538, 6.2085, 77.5, 3.2074499999999997, 5.0, 330.0, 19.05, 391.44, 11.36, 21.2], ['75%', 3.6770824999999996, 12.5, 18.1, 0.0, 0.624, 6.6235, 94.07499999999999, 5.1884250000000005, 24.0, 666.0, 20.2, 396.225, 16.955000000000002, 25.0], ['max', 88.9762, 100.0, 27.74, 1.0, 0.871, 8.78, 100.0, 12.1265, 24.0, 711.0, 22.0, 396.9, 37.97, 50.0]]
        assert B.dataset.shape == shape, " TEST FAILED: Failed to verify dataset shape"
        assert str(B.dataset.dtypes) == dtypes, " TEST FAILED: Failed to verify dataset dtypes string"
        assert B.dataset.describe().reset_index().values.tolist() == description, " TEST FAILED: Failed to verify dataset description"
    def test_data_visualizations(self):
        assert os.path.isfile("box_and_whisker_plot.png") == True, " TEST FAILED: Failed to verify file path: box_and_whisker_plot.png"
        assert os.path.isfile("correlation_matrix.png") == True, " TEST FAILED: Failed to verify file path: correlation_matrix.png"
        assert os.path.isfile("density_plot.png") == True, " TEST FAILED: Failed to verify file path: density_plot.png"
        assert os.path.isfile("histogram.png") == True, " TEST FAILED: Failed to verify file path: histogram.png"
        assert os.path.isfile("scatter_plot.png") == True, " TEST FAILED: Failed to verify file path: scatter_plot.png"
    def test_partioning_dataset(self):
        assert len(B.x) == 506, " TEST FAILED: Failed to verify size of: X dataset"
        assert len(B.y) == 506, " TEST FAILED: Failed to verify size of: Y dataset"
        assert len(B.x_train) == 404, " TEST FAILED: Failed to verify size of: x_train dataset"
        assert len(B.x_validation) == 102, " TEST FAILED: Failed to verify size of: x_validation dataset"
        assert len(B.y_train) == 404, " TEST FAILED: Failed to verify size of: y_train dataset"
        assert len(B.y_validation) == 102, " TEST FAILED: Failed to verify size of: y_validation dataset"