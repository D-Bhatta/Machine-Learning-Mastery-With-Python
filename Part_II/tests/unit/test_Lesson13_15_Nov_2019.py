#testcases for Lesson13_15_Nov_2019.py
import sys
import temp
from numpy import array
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson13 import Lesson13_15_Nov_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data(self):
        x,y = ls.load_data()
        x = str(x)
        y = str(y)
        temp_x,temp_y = temp.temp_load_x_pima(),temp.temp_load_y_pima()
        assert (x,y == temp_x,temp_y), "Couldn't load Pima data x and y values at load_data()"
    
    '''def test_create_pipeline_data_preparation(self):
        model = ls.create_pipeline_data_preparation
        assert str(model) == "Pipeline(memory=None,\n         steps=[('standardize',\n                 StandardScaler(copy=True, with_mean=True, with_std=True)),\n                ('lda',\n                 LinearDiscriminantAnalysis(n_components=None, priors=None,\n                                            shrinkage=None, solver='svd',\n                                            store_covariance=False,\n                                            tol=0.0001))],\n         verbose=False)",        "The model string doesn't match the anticipated model string at pipeline_data_preparation()"'''

    def test_evaluate_pipeline_data_preparation(self):
        results = ls.evaluate_pipeline_data_preparation()
        assert results == 0.773462064251538, "The mean of the cross Validation score results doesn't match anticipated value at evaluate_pipeline_data_preparation()"

    '''def test_create_pipeline_feature_extraction(self):
        model = ls.create_pipeline_feature_extraction()
        assert str(model) == "Pipeline(memory=None,\n         steps=[('feature_union',\n                 FeatureUnion(n_jobs=None,\n                              transformer_list=[('pca',\n                                                 PCA(copy=True,\n                                                     iterated_power='auto',\n                                                     n_components=3,\n                                                     random_state=None,\n                                                     svd_solver='auto', tol=0.0,\n                                                     whiten=False)),\n                                                ('select_best',\n                                                 SelectKBest(k=6,\n                                                             score_func=<function f_classif at 0x00000209BB65B9D8>))],\n                              transformer_weights=None, verbose=False)),\n                ('logistic',\n                 LogisticRegression(C=1.0, class_weight=None, dual=False,\n                                    fit_intercept=True, intercept_scaling=1,\n                                    l1_ratio=None, max_iter=100,\n                                    multi_class='warn', n_jobs=None,\n                                    penalty='l2', random_state=None,\n                                    solver='warn', tol=0.0001, verbose=0,\n                                    warm_start=False))],\n         verbose=False)",        "The model string doesn't match the anticipated model string at pipeline_feature_extraction()"'''

    def test_evaluate_pipeline_feature_extraction(self):
        results = ls.evaluate_pipeline_feature_extraction()
        assert results == 0.7760423786739576, "The mean of the cross Validation score results doesn't match anticipated value at evaluate_pipeline_feature_extraction()"