# Python Project Template
# import warnings filter
from warnings import simplefilter

# Logging setup
import logging
import logging.config
from json import load as jload

# 1. Prepare Problem
""" This step is about loading everything you need to start working on your problem. This is also the home of any global configuration you might need to do. It is also the place
where you might need to make a reduced sample of your dataset if it is too large to work with. """
# a) Load libraries
import numpy
from matplotlib import pyplot as plt
from pandas import read_csv, set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
""" Configure logger lg with config for appLogger from config.json["logging"] """
with open('config.json', 'r') as f:
        config = jload(f)
        logging.config.dictConfig(config["logging"])
lg = logging.getLogger('appLogger')
# lg.debug("This is a debug message")

class Sonar(object):
    def __init__(self):
        # b) Load dataset
        url = 'sonar/sonar.all-data.csv'
        self.data = read_csv(url, header=None)
        self.array = self.data.values
        self.x = self.array[:,0:60].astype(float)
        self.y = self.array[:,60]
        validation_size = 0.20
        seed = 15
        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(self.x, self.y, test_size=validation_size, random_state=seed)
    
    # 2. Summarize Data
    """ This step is about better understanding the data that you have available. This includes
    understanding your data using

    - Descriptive statistics such as summaries.
    - Data visualizations such as plots with Matplotlib, ideally using convenience functions from
    Pandas.

    Take your time and use the results to prompt a lot of questions, assumptions and hypotheses
    that you can investigate later with specialized models. """
    # a) Descriptive statistics
    def descriptive_statistics(self,prn):
        """ 
        Descriptive statistics about the dataset. 
        input:
        prn:
            print stats about the data
            boolean
        Output:
        prints stats about the data 
        returns a tuple of data stat strings """
        
        stat_strings = [("Shape of dataset: \n",self.data.shape),("Dtypes: \n",self.data.dtypes),("Data(20 rows): \n", self.data.head(20)),("Data description: \n",self.data.describe()),("Class disributions: \n",self.data.groupby(60).size())]
        if(prn):
            # set number of rows to display
            set_option('display.max_rows', 500)
            # set number of pixels cols should occupy
            set_option('display.width', 100)
            # set precision of values
            set_option('precision', 2)
            for i in stat_strings:
                print(i[0],i[1])


    def unimodal_viz(self):
        self.data.hist(sharex = False, sharey = False, xlabelsize = 0.5, ylabelsize=0.5)
        plt.savefig('histogram.png', format='png')
        self.data.plot(kind='density', subplots=True, layout=(8,8), sharex=False,legend=False, fontsize=1, sharey=False)
        plt.savefig('density_plot.png', format='png')
        """ There seems to be a pandas bug here """
        # self.data.plot(kind='box', subplots=True, sharex=False, fontsize=1)
        # plt.savefig('box_plot.png', format='png')
    def multimodal_viz(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.data.corr(), vmin = -1, vmax=1, interpolation='none')
        fig.colorbar(cax)
        plt.savefig('correlation_matrix.png', format='png')
        pass
    # b) Data visualizations
    def data_visualizetions(self):
        """ 
        Visualizes the data in the dataset
        """
        self.unimodal_viz()
        self.multimodal_viz()
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

sonar = Sonar()
# sonar.descriptive_statistics(prn=True)
sonar.data_visualizetions()