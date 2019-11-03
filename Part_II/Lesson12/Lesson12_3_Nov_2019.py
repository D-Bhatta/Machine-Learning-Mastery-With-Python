#Lesson 12
#
# Page : 92/179
#
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
from pandas import read_csv
#change the last folder name
os.chdir("J:\Education\Code\DATA_Science\Books\Jason_Brownlee\Machine-Learning-Mastery-With-Python\Part_II\Lesson12") # pylint: disable=anomalous-backslash-in-string