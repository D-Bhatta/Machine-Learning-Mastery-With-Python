from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from numpy import array
# Python Project: Iris Classification
class iris(object):
    # 1. Prepare Problem: Load dataset and libraries
    def __init__(self):        
        """ We will be loading 'iris.data' in a pandas dataframe and loading libraries and variables here"""
        # a) Load libraries above
        # b) Load dataset
        filename = 'iris.data'
        names = ['sepal-lenght', 'sepal-width', 'petal-length', 'petal-width', 'class']
        self.dataset = read_csv(filename, names=names)
        self.x = []
        self.y = []
        self.x_train, self.y_train, self.x_validation, self.y_validation = [],[],[],[]
    # 2. Summarize Data
    def summarize_iris_data(self):
        """ This step is about better understanding the data that you have available. This includes
        understanding your data using

        - Descriptive statistics such as summaries.
        - Data visualizations such as plots with Matplotlib, ideally using convenience functions from
        Pandas.

        Take your time and use the results to prompt a lot of questions, assumptions and hypotheses
        that you can investigate later with specialized models. """
        # a) Descriptive statistics
        def summarize_iris_data_stats():         
            shape = self.dataset.shape
            """ We can see that there are 150 instances(or rows) and 5 attributes """
            head = self.dataset.head(20)  
            """ A look at the first 20 rows shows us that The data X values are of ratio(float) type and the y values are categorical and nominal """
            statistical_summary = self.dataset.describe()
            """ From the summary we can see that the data is of 150 count. The values lie between 0 and 8. """
            class_distribution = self.dataset.groupby('class').size()
            """ We can see that the class distributions are well balanced, with each of the 3 classes comprising a neat third of the dataset."""
            print("Shape of the dataset(instance,attribute): ",shape,""" We can see that there are 150 instances(or rows) and 5 attributes """,
            "First 20 instances: ",head,""" A look at the first 20 rows shows us that The data X values are of ratio(float) type and the y values are categorical and nominal """,
            "Statistical summary: ",statistical_summary,""" From the summary we can see that the data is of 150 count. The values lie between 0 and 8. """,
            "Class Distribution: ",class_distribution,""" We can see that the class distributions are well balanced, with each of the 3 classes comprising a neat third of the dataset.""",
            sep='\n')  
        summarize_iris_data_stats() 
        # b) Data visualizations
        def summarize_iris_data_visualtization(): 
            # box-plot
            self.dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) 
            plt.savefig("box_and_whisker_plot.png", format = 'png')#saves the plot
            # histogram
            self.dataset.hist()
            plt.savefig("histogram_plot.png", format = 'png')#saves the plot
            # Scatter matrix
            iris_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      
             0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,       
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,       
             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,       
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,       
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]# Targets the iris dataset
            color_wheel = {1: "#0392cf", 
                        2: "#7bc043", 
                        3: "#ee4035"}#colors for each of the classes
            colors = list(map(lambda x: color_wheel.get(x + 1), iris_target))#map colors to classes
            scatter_matrix(self.dataset, color=colors)#create the scatter matrix
            plt.savefig("scatter_matrix.png", format = 'png')#saves the plot
            print(" Data Visualiztion & analysis\nBox and whisker\n\nSepal length\nWe can see a well balanced dataset. There is no visible skew. The max data point seems to be well above the 75% quartile.\nSepal width\nWe can see some outliers here, above the max point. There is slight skew towards the 75% quartile and, the data is probably skewed to the right.\nPetal length\nNo outliers, but the data is very much skewed towards the 25% quartile. The 75% quartile is much closer to the mean than the 25% quartile. The minimum value is quite far from the mean.\nPetal width\nAgain, the data is very much skewed towards the 25% quartile. The minimum value is quite far from the mean. \nConclusion\nPetal length and width are both on the smaller side. Values in these 2 columns are skewed to the left. Very interesting.\nIn contrast, sepal length and width are much more 'normal'.\n\nHistogram\n\nAs expected, petal length and width are both heavily skewed to the left. You could draw a diagonal line from the left to the right across the Maximas of the petal width data.\nSepal length and width assume a very broken, but still imaginable bell curve.\nOverall, the data seems very interesting.\n\nScatter matrix\n\nThere's a slight correlation between sepal length and sepal width for one of the classes. This is also the case for sepal length and petal length.\nPetal length and width also have a correlation for a part of the data.\nConclusion\nThe data has some slight correlation. ")
        summarize_iris_data_visualtization()
            
    def evaluate_algorithms(self):
        # 4. Evaluate Algorithms
        """ This step is about finding a subset of machine learning algorithms that are good at exploiting
        the structure of your data (e.g. have better than average skill).
        On a given problem you will likely spend most of your time on this and the previous step
        until you converge on a set of 3-to-5 well performing machine learning algorithms. """
        # a) Split-out validation dataset
        def partition_data():
            array = self.dataset.values
            self.x = array[:,0:4]
            self.y = array[:,4]
            validation_size = 0.20
            random_seed = 7
            self.x_train,self.x_validation,self.y_train,self.y_validation = train_test_split(self.x,self.y,test_size=validation_size,random_state=random_seed)
        partition_data()
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

iris = iris()
iris.summarize_iris_data()
""" output:
Shape of the dataset(instance,attribute):
(150, 5)
 We can see that there are 150 instances(or rows) and 5 attributes
First 20 instances:
    sepal-lenght  sepal-width  petal-length  petal-width        class
0            5.1          3.5           1.4          0.2  Iris-setosa
1            4.9          3.0           1.4          0.2  Iris-setosa
2            4.7          3.2           1.3          0.2  Iris-setosa
3            4.6          3.1           1.5          0.2  Iris-setosa
4            5.0          3.6           1.4          0.2  Iris-setosa
5            5.4          3.9           1.7          0.4  Iris-setosa
6            4.6          3.4           1.4          0.3  Iris-setosa
7            5.0          3.4           1.5          0.2  Iris-setosa
8            4.4          2.9           1.4          0.2  Iris-setosa
9            4.9          3.1           1.5          0.1  Iris-setosa
10           5.4          3.7           1.5          0.2  Iris-setosa
11           4.8          3.4           1.6          0.2  Iris-setosa
12           4.8          3.0           1.4          0.1  Iris-setosa
13           4.3          3.0           1.1          0.1  Iris-setosa
14           5.8          4.0           1.2          0.2  Iris-setosa
15           5.7          4.4           1.5          0.4  Iris-setosa
16           5.4          3.9           1.3          0.4  Iris-setosa
17           5.1          3.5           1.4          0.3  Iris-setosa
18           5.7          3.8           1.7          0.3  Iris-setosa
19           5.1          3.8           1.5          0.3  Iris-setosa
 A look at the first 20 rows shows us that The data X values are of ratio(float) type and the y values are categorical and nominal
Statistical summary:
       sepal-lenght  sepal-width  petal-length  petal-width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
 From the summary we can see that the data is of 150 count. The values lie between 0 and 8.
Class Distribution:
class
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
dtype: int64
 We can see that the class distributions are well balanced, with each of the 3 classes comprising a neat third of the dataset.
 Data Visualiztion & analysis
Box and whisker

Sepal length
We can see a well balanced dataset. There is no visible skew. The max data point seems to be well above the 75% quartile.
Sepal width
We can see some outliers here, above the max point. There is slight skew towards the 75% quartile and, the data is probably skewed to the right.
Petal length
No outliers, but the data is very much skewed towards the 25% quartile. The 75% quartile is much closer to the mean than the 25% quartile. The minimum value is quite far from the mean.
Petal width
Again, the data is very much skewed towards the 25% quartile. The minimum value is quite far from the mean.
Conclusion
Petal length and width are both on the smaller side. Values in these 2 columns are skewed to the left. Very interesting.
In contrast, sepal length and width are much more 'normal'.

Histogram

As expected, petal length and width are both heavily skewed to the left. You could draw a diagonal line from the left to the right across the Maximas of the petal width data.
Sepal length and width assume a very broken, but still imaginable bell curve.
Overall, the data seems very interesting.

Scatter matrix

There's a slight correlation between sepal length and sepal width for one of the classes. This is also the case for sepal length and petal length.
Petal length and width also have a correlation for a part of the data.
Conclusion
The data has some slight correlation.
"""
