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
        self.names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT', 'MEDV']
        # Load the dataset into a class data member
        try:
            self.dataset = read_csv(filename, delim_whitespace=True, names=self.names)
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
        """ We will now take a closer look at the data and analyze it using descriptive statistics and data visualizations."""
        # a) Descriptive statistics
        def descriptive_statistics():
            # Shape of the dateset
            shape = self.dataset.shape
            """ We can see that there are 506 instances or rows and 11 attributes or columns. """
            # Datatype of the attributes
            dtypes = self.dataset.dtypes
            # Peek at the first 20 rows of the data
            head = self.dataset.head(20)
            # Summarize attribute distribution
            set_option('precision', 1)
            description = self.dataset.describe()
            # Look at the correlation between the different attributes
            set_option('precision',2)
            correlation = self.dataset.corr(method='pearson')
            # Print the results
            print("The shape of the dataset is {}. We can see that there are 506 instances or rows and 11 attributes or columns.".format(shape))
            print("The datatpes of the attributes are: \n{}.\nMost of them are float types, with 2 integer types mixed in.".format(dtypes))
            print("Lets take a look at the first 20 rows of the data:\n{}.\nWe can see that all the data is disparate from each other, with dissimilar ranges. Some attributes are binary(0 & 1). Some have a lot of 0s in them.".format(head))
            print("Dataset description:\n{}".format(description),"\nWe can see that the attributes have wildly varying ranges, means and min and max values. The data needs to be brought to scale. Also, the ZN attribute might be significant.")
            print("Correlation between the different attributes:\n{}".format(correlation),
            """\nWe can see that many of the attributes have high correlation among them. 
            DIS and NOX at 0.77 and RAD and TAX with 0.91 certainly jump out. CHAS seems the least correlated to other attributes.""")
        descriptive_statistics()    
        # b) Data visualizations
        def data_visualization():
            """ We will now visaualize our data and try to analyse it and gain a better understanding of the data. """
            def histogram():
                self.dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
                #save the image to a file
                plt.savefig('histogram.png', format='png')
            def density_plot():
                self.dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False, legend=False,fontsize=1)
                #save the image to a file
                plt.savefig('density_plot.png', format='png')
            def box_and_whisker():
                self.dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False, fontsize=8)
                #save the image to a file
                plt.savefig('box_and_whisker_plot.png', format='png')
            histogram()
            density_plot()
            box_and_whisker()
            print("\n\nData Visualization:")
            print("""The histogram depicts each of the attributes in a separate histogram. \nI think we can see that almost all of the attbutes except 'RM' are heavily skewed. The data would probably benefit from scaling anf transformation.\nA lot of the attributes might be bimodal.""")
            print("""The density plot shows a clearer picture of the distributions. Clearly there is a lot of skew and there are a lot of bimodal distributions. """)
            print("""The box and whisker plots paint a very clear picture of skewness of the data, as well as of the bimodial distributions. """)
            def scatter_plot():
                scatter_matrix(self.dataset)
                plt.savefig("scatter_plot.png", format='png')
            def correlation_matrix():
                fig = plt.figure()
                ax = fig.add_subplot(111)
                cax = ax.matshow(self.dataset.corr(), vmin=-1, vmax= 1, interpolation='none')
                fig.colorbar(cax)
                ticks = numpy.arange(0,14,1)
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(self.names)
                ax.set_yticklabels(self.names)
                plt.savefig('correlation_matrix.png', format='png')
            scatter_plot()
            correlation_matrix()
            print(""" \nCorrelation:\nWe can see a lot of negative and positive correlation in the form of nice and smooth curves in the scatter plot. """)
            print(""" \nThe correlation matrix confirms the correlation evident in the scatter plot. DIS seems to be highly correlated with no less than 4 variables. AGE is also correlated, and this dataset would benefit from being standardized and perhaps some PCA. """)
        data_visualization()
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
The shape of the dataset is (506, 14). We can see that there are 506 instances or rows and 11 attributes or columns.
The datatpes of the attributes are:
CRIM       float64
ZN         float64
INDUS      float64
CHAS         int64
NOX        float64
RM         float64
AGE        float64
DIS        float64
RAD          int64
TAX        float64
PTRATIO    float64
B          float64
LSTAT      float64
MEDV       float64
dtype: object.
Most of them are float types, with 2 integer types mixed in.
Lets take a look at the first 20 rows of the data:
        CRIM    ZN  INDUS  CHAS   NOX    RM    AGE   DIS  RAD    TAX  PTRATIO       B  LSTAT  MEDV
0   6.32e-03  18.0   2.31     0  0.54  6.58   65.2  4.09    1  296.0     15.3  396.90   4.98  24.0
1   2.73e-02   0.0   7.07     0  0.47  6.42   78.9  4.97    2  242.0     17.8  396.90   9.14  21.6
2   2.73e-02   0.0   7.07     0  0.47  7.18   61.1  4.97    2  242.0     17.8  392.83   4.03  34.7
3   3.24e-02   0.0   2.18     0  0.46  7.00   45.8  6.06    3  222.0     18.7  394.63   2.94  33.4
4   6.91e-02   0.0   2.18     0  0.46  7.15   54.2  6.06    3  222.0     18.7  396.90   5.33  36.2
5   2.99e-02   0.0   2.18     0  0.46  6.43   58.7  6.06    3  222.0     18.7  394.12   5.21  28.7
6   8.83e-02  12.5   7.87     0  0.52  6.01   66.6  5.56    5  311.0     15.2  395.60  12.43  22.9
7   1.45e-01  12.5   7.87     0  0.52  6.17   96.1  5.95    5  311.0     15.2  396.90  19.15  27.1
8   2.11e-01  12.5   7.87     0  0.52  5.63  100.0  6.08    5  311.0     15.2  386.63  29.93  16.5
9   1.70e-01  12.5   7.87     0  0.52  6.00   85.9  6.59    5  311.0     15.2  386.71  17.10  18.9
10  2.25e-01  12.5   7.87     0  0.52  6.38   94.3  6.35    5  311.0     15.2  392.52  20.45  15.0
11  1.17e-01  12.5   7.87     0  0.52  6.01   82.9  6.23    5  311.0     15.2  396.90  13.27  18.9
12  9.38e-02  12.5   7.87     0  0.52  5.89   39.0  5.45    5  311.0     15.2  390.50  15.71  21.7
13  6.30e-01   0.0   8.14     0  0.54  5.95   61.8  4.71    4  307.0     21.0  396.90   8.26  20.4
14  6.38e-01   0.0   8.14     0  0.54  6.10   84.5  4.46    4  307.0     21.0  380.02  10.26  18.2
15  6.27e-01   0.0   8.14     0  0.54  5.83   56.5  4.50    4  307.0     21.0  395.62   8.47  19.9
16  1.05e+00   0.0   8.14     0  0.54  5.93   29.3  4.50    4  307.0     21.0  386.85   6.58  23.1
17  7.84e-01   0.0   8.14     0  0.54  5.99   81.7  4.26    4  307.0     21.0  386.75  14.67  17.5
18  8.03e-01   0.0   8.14     0  0.54  5.46   36.6  3.80    4  307.0     21.0  288.99  11.69  20.2
19  7.26e-01   0.0   8.14     0  0.54  5.73   69.5  3.80    4  307.0     21.0  390.95  11.28  18.2.
We can see that all the data is disparate from each other, with dissimilar ranges. Some attributes are binary(0 & 1). Some have a lot of 0s in them.
Dataset description:
           CRIM      ZN   INDUS    CHAS     NOX      RM  ...     RAD     TAX  PTRATIO       B   LSTAT    MEDV
count  5.06e+02  506.00  506.00  506.00  506.00  506.00  ...  506.00  506.00   506.00  506.00  506.00  506.00
mean   3.61e+00   11.36   11.14    0.07    0.55    6.28  ...    9.55  408.24    18.46  356.67   12.65   22.53
std    8.60e+00   23.32    6.86    0.25    0.12    0.70  ...    8.71  168.54     2.16   91.29    7.14    9.20
min    6.32e-03    0.00    0.46    0.00    0.39    3.56  ...    1.00  187.00    12.60    0.32    1.73    5.00
25%    8.20e-02    0.00    5.19    0.00    0.45    5.89  ...    4.00  279.00    17.40  375.38    6.95   17.02
50%    2.57e-01    0.00    9.69    0.00    0.54    6.21  ...    5.00  330.00    19.05  391.44   11.36   21.20
75%    3.68e+00   12.50   18.10    0.00    0.62    6.62  ...   24.00  666.00    20.20  396.23   16.96   25.00
max    8.90e+01  100.00   27.74    1.00    0.87    8.78  ...   24.00  711.00    22.00  396.90   37.97   50.00

[8 rows x 14 columns]
We can see that the attributes have wildly varying ranges, means and min and max values. The data needs to be brought to scale. Also, the ZN attribute might be significant.
Correlation between the different attributes:
         CRIM    ZN  INDUS      CHAS   NOX    RM   AGE   DIS       RAD   TAX  PTRATIO     B  LSTAT  MEDV
CRIM     1.00 -0.20   0.41 -5.59e-02  0.42 -0.22  0.35 -0.38  6.26e-01  0.58     0.29 -0.39   0.46 -0.39
ZN      -0.20  1.00  -0.53 -4.27e-02 -0.52  0.31 -0.57  0.66 -3.12e-01 -0.31    -0.39  0.18  -0.41  0.36
INDUS    0.41 -0.53   1.00  6.29e-02  0.76 -0.39  0.64 -0.71  5.95e-01  0.72     0.38 -0.36   0.60 -0.48
CHAS    -0.06 -0.04   0.06  1.00e+00  0.09  0.09  0.09 -0.10 -7.37e-03 -0.04    -0.12  0.05  -0.05  0.18
NOX      0.42 -0.52   0.76  9.12e-02  1.00 -0.30  0.73 -0.77  6.11e-01  0.67     0.19 -0.38   0.59 -0.43
RM      -0.22  0.31  -0.39  9.13e-02 -0.30  1.00 -0.24  0.21 -2.10e-01 -0.29    -0.36  0.13  -0.61  0.70
AGE      0.35 -0.57   0.64  8.65e-02  0.73 -0.24  1.00 -0.75  4.56e-01  0.51     0.26 -0.27   0.60 -0.38
DIS     -0.38  0.66  -0.71 -9.92e-02 -0.77  0.21 -0.75  1.00 -4.95e-01 -0.53    -0.23  0.29  -0.50  0.25
RAD      0.63 -0.31   0.60 -7.37e-03  0.61 -0.21  0.46 -0.49  1.00e+00  0.91     0.46 -0.44   0.49 -0.38
TAX      0.58 -0.31   0.72 -3.56e-02  0.67 -0.29  0.51 -0.53  9.10e-01  1.00     0.46 -0.44   0.54 -0.47
PTRATIO  0.29 -0.39   0.38 -1.22e-01  0.19 -0.36  0.26 -0.23  4.65e-01  0.46     1.00 -0.18   0.37 -0.51
B       -0.39  0.18  -0.36  4.88e-02 -0.38  0.13 -0.27  0.29 -4.44e-01 -0.44    -0.18  1.00  -0.37  0.33
LSTAT    0.46 -0.41   0.60 -5.39e-02  0.59 -0.61  0.60 -0.50  4.89e-01  0.54     0.37 -0.37   1.00 -0.74
MEDV    -0.39  0.36  -0.48  1.75e-01 -0.43  0.70 -0.38  0.25 -3.82e-01 -0.47    -0.51  0.33  -0.74  1.00
We can see that many of the attributes have high correlation among them.
            DIS and NOX at 0.77 and RAD and TAX with 0.91 certainly jump out. CHAS seems the least correlated to other attributes.
 """