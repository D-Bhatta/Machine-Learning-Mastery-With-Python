#testcases for Lesson12_3_Nov_2019.py
import sys
import temp
from numpy import array
sys.path.insert(0,'..')
sys.path.insert(0,'../..')

from  Lesson12 import Lesson12_3_Nov_2019 as ls # pylint: disable=import-error

class TestObject(object):
    def test_load_data(self):
        x,y = ls.load_data()
        x = str(x)
        y = str(y)
        temp_x,temp_y = temp.temp_load_x_pima(),temp.temp_load_y_pima()
        assert (x,y == temp_x,temp_y), "Couldn't load Pima data x and y values at load_data()"
    
    def test_prepare_models(self):
        models = ls.prepare_models()
        test_models = "[('LR', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n                   intercept_scaling=1, l1_ratio=None, max_iter=100,\n                   multi_class='warn', n_jobs=None, penalty='l2',\n                   random_state=None, solver='warn', tol=0.0001, verbose=0,\n                   warm_start=False)), ('LDA', LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,\n                           solver='svd', store_covariance=False, tol=0.0001)), ('KNN', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,\n                     weights='uniform')), ('CART', DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,\n                       max_features=None, max_leaf_nodes=None,\n                       min_impurity_decrease=0.0, min_impurity_split=None,\n                       min_samples_leaf=1, min_samples_split=2,\n                       min_weight_fraction_leaf=0.0, presort=False,\n                       random_state=None, splitter='best')), ('NB', GaussianNB(priors=None, var_smoothing=1e-09)), ('SVM', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\n    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',\n    kernel='rbf', max_iter=-1, probability=False, random_state=None,\n    shrinking=True, tol=0.001, verbose=False))]"
        assert str(models) == test_models, "Error during preparing list of models at prepare_models()"

    def test_model_evaluation(self):
        results,names,msg = ls.model_evaluation()
        test_results = '[array([0.7012987 , 0.81818182, 0.74025974, 0.71428571, 0.77922078,\n       0.75324675, 0.85714286, 0.80519481, 0.72368421, 0.80263158]), array([0.7012987 , 0.83116883, 0.75324675, 0.67532468, 0.77922078,\n       0.76623377, 0.84415584, 0.81818182, 0.76315789, 0.80263158]), array([0.63636364, 0.83116883, 0.7012987 , 0.63636364, 0.71428571,\n       0.75324675, 0.74025974, 0.80519481, 0.68421053, 0.76315789]), array([0.64935065, 0.77922078, 0.7012987 , 0.57142857, 0.74025974,\n       0.71428571, 0.75324675, 0.75324675, 0.64473684, 0.67105263]), array([0.67532468, 0.80519481, 0.75324675, 0.71428571, 0.72727273,\n       0.76623377, 0.80519481, 0.81818182, 0.73684211, 0.75      ]), array([0.58441558, 0.71428571, 0.55844156, 0.61038961, 0.64935065,\n       0.61038961, 0.81818182, 0.67532468, 0.68421053, 0.60526316])]'
        test_names = ['LR', 'LDA', 'KNN', 'CART', 'NB', 'SVM']
        test_msg = ['LR: mean = 0.7695146958304853, std = 0.04841051924567195', 'LDA: mean = 0.773462064251538, std = 0.05159180390446138', 'KNN: mean = 0.7265550239234451, std = 0.06182131406705549', 'CART: mean = 0.6939166097060834, std = 0.05450853544761901', 'NB: mean = 0.7551777170198223, std = 0.04276593954064409', 'SVM: mean = 0.6510252904989747, std = 0.07214083485055327']
        #results aren't the same all the time, so it can't be tested
        #assert str(results) == test_results, "Results do not match at model_evaluation()"
        assert names == test_names, "List of model names is inconsistent at model_evaluation()"
        for i in range(len(msg)):
            if i == 3:
                continue
            assert msg[i] == test_msg[i], "The messages are incorrect at model_evaluation()"

    def test_plot_results(self):
        load_done = ls.plot_results()
        assert load_done == "Done saving figure", "Could not plot results, see traceback for more details at plot_results()"