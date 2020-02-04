# Python Project Template
# 1. Prepare Problem
""" This step is about loading everything you need to start working on your problem. This is also the home of any global configuration you might need to do. It is also the place
where you might need to make a reduced sample of your dataset if it is too large to work with. """
# a) Load libraries
import numpy
from numpy import arange
from matplotlib import pyplot as plt
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import  AdaBoostRegressor
from sklearn.metrics import mean_squared_error
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Logging setup
import logging
import logging.config
from json import load as jload
""" Configure logger lg with config for appLogger from config.json["logging"] """
with open('config.json', 'r') as f:
        config = jload(f)
        logging.config.dictConfig(config["logging"])
lg = logging.getLogger('appLogger')
# lg.debug("This is a debug message")


class Boston(object):
    """Class definition for the boston housing dataset analysis"""
    # b) Load dataset
    def __init__(self):
        """ We will be loading the Boston Housing Dataset into a pandas dataframe here, and then load class variables"""
        filename = "housing.data"
        names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',]
        # Load the dataset into a class data member
        try:
            self.dataset = read_csv(filename, delim_whitespace=True, names=names)
        except FileNotFoundError:
            lg.error("FIle not found error", exc_info=True)
            exit(1)
        # Check that the dataset hs been loaded properly
        assert self.dataset.size != 0 , "The data hasn't been loaded correctly"
    # 2. Summarize Data
    """ This step is about better understanding the data that you have available. This includes
    understanding your data using

    - Descriptive statistics such as summaries.
    - Data visualizations such as plots with Matplotlib, ideally using convenience functions from
    Pandas.

    Take your time and use the results to prompt a lot of questions, assumptions and hypotheses
    that you can investigate later with specialized models. """
    def analyze_data(self):
        """ We will now take a closer look at the data and analyze it using descriptive statistics."""
        def descriptive_statistics():
            # Shape of the dateset
            shape = self.dataset.shape
            """ We can see that there are 506 instances or rows and 11 attributes or columns. """
            # Print the results
            print("The shape of the dataset is {}. We can see that there are 506 instances or rows and 11 attributes or columns.".format(shape))

        descriptive_statistics()
    # a) Descriptive statistics
    # b) Data visualizations
    # 3. Prepare Data
    """ This step is about preparing the data in such a way that it best exposes the structure of the
    problem and the relationships between your input attributes with the output variable.
    Start simple. Revisit this step often and cycle with the next step until you converge on a
    subset of algorithms and a presentation of the data that results in accurate or accurate-enough
    models to proceed. """
    # a) Data Cleaning
    # b) Feature Selection
    # c) Data Transforms
    # 4. Evaluate Algorithms
    """ This step is about finding a subset of machine learning algorithms that are good at exploiting
    the structure of your data (e.g. have better than average skill).
    On a given problem you will likely spend most of your time on this and the previous step
    until you converge on a set of 3-to-5 well performing machine learning algorithms. """
    # a) Split-out validation dataset
    # b) Test options and evaluation metric
    # c) Spot Check Algorithms
    # d) Compare Algorithms
    # 5. Improve Accuracy
    """ Once you have a shortlist of machine learning algorithms, you need to get the most out of them.
    The line between this and the previous step can blur when a project becomes concrete.
    There may be a little algorithm tuning in the previous step. And in the case of ensembles, you
    may bring more than a shortlist of algorithms forward to combine their predictions. """
    # a) Algorithm Tuning
    # b) Ensembles
    # 6. Finalize Model
    """ Once you have found a model that you believe can make accurate predictions on unseen data,
    you are ready to finalize it. """
    # a) Predictions on validation dataset
    # b) Create standalone model on entire training dataset
    # c) Save model for later use
    """ ### Tips for using the template

    - Fast First Pass . Make a first-pass through the project steps as fast as possible. This
    will give you confidence that you have all the parts that you need and a baseline from
    which to improve.
    - Cycles . The process in not linear but cyclic. You will loop between steps, and probably
    spend most of your time in tight loops between steps 3-4 or 3-4-5 until you achieve a level
    of accuracy that is sufficient or you run out of time.
    - Attempt Every Step . It is easy to skip steps, especially if you are not confident or
    familiar with the tasks of that step. Try and do something at each step in the process,
    even if it does not improve accuracy. You can always build upon it later. Don’t skip steps,
    just reduce their contribution.
    - Ratchet Accuracy . The goal of the project is model accuracy. Every step contributes
    towards this goal. Treat changes that you make as experiments that increase accuracy as
    the golden path in the process and reorganize other steps around them. Accuracy is a
    ratchet that can only move in one direction (better, not worse).
    - Adapt As Needed . Modify the steps as you need on a project, especially as you become
    more experienced with the template. Blur the edges of tasks, such as steps 4-5 to best
    serve model accuracy. """


boston = Boston()
boston.analyze_data()

""" Output:
The shape of the dataset is (506, 11). We can see that there are 506 instances or rows and 11 attributes or columns.
 """