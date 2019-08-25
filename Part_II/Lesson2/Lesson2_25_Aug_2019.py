#Lesson 2
#
# Page : 23/179
#
#Strings
def string_example():
    data = 'hello world'
    print(data[0])
    print(len(data))
    print(data)
#Numbers
def numbers_example():
    value = 123.1
    print(value)
    value = 10
    print(value)
#Boolean
def boolean_example():
    a = True
    b = False
    print(a,b)
#Multiple Assignment
def multiple_assignment_example():
    a,b,c = 1,2,3
    print(a, b, c)
#No value
def no_value_example():
    a = None
    print(a)
#Tuple
def tuple_example():
    a = (1,2,3)
    print(a)
#dictionary
def dictionary_example():
    my_dict = {'a':1,'b':2,'c':3}
    print("A value:{}".format(my_dict['a']))
    my_dict['a'] = 100
    print("A value:{}".format(my_dict['a']))
    print("keys:{}".format(my_dict.keys()))
    print("Values:{}".format(my_dict.values()))
    for key in my_dict.keys(): print(my_dict[key])
#Numpy Crash Course
import numpy
#create an array
def numpy_array_example():
    mylist = [1,2.3]
    myarray = numpy.array(mylist)
    print(myarray)
    print(myarray.shape)#shape returns a tuple with the n,m,k,j.... where n is the number of rows, m is the number of columns, and so on
#access data
def numpy_data_access_example():
    mylist = [[1,2,3],[3,4,5]]
    myarray  = numpy.array(mylist)
    print(myarray)
    print(myarray.shape)
    print("first row:{}\nlast row:{}".format(myarray[0],myarray[-1]))
    print("specific:{}".format(myarray[0,2]))
    print("col:{}".format(myarray[:2]))
#arithmatic
def numpy_arithmatic_example(list1,list2):
    myarray1 = numpy.array(list1)
    myarray2 = numpy.array(list2)
    return myarray1+myarray2,myarray1*myarray2#(array([5, 5, 5]), array([6, 6, 6]))
#
#matplotlib crahs course
import matplotlib.pyplot as plt
#line plot
def line_plot_example():
    myarr = numpy.array([1,2,3])
    plt.plot(myarr)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()
#scatter plot example
def scatter_plot_example():
    x = numpy.array([1,2,3])
    y = numpy.array([2,4,6])
    plt.scatter(x,y)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()
#pandas crash course
#
import pandas
#series example
def pandas_series_example():
    myarr = numpy.array([1,2,3])
    rownames = ['a','b','c']
    myseries = pandas.Series(myarr, index = rownames)
    '''a    1
    b    2
    c    3'''
    return myseries
#data access
def pandas_series_data_access_example(myseries):
    a = myseries[0]#1
    b = myseries['a']#1
    return a,b#(1, 1)
#dataframe example
def pandas_dataframe_example():
    myarr = numpy.array([[1,2,3],[4,5,6]])
    rownames = ['a','b']
    colnames = ['one','two','three']
    mydataframe = pandas.DataFrame(myarr, index = rownames,columns=colnames)
    '''   one  two  three
    a    1    2      3
    b    4    5      6'''
    return mydataframe
#data access
def pandas_dataframe_data_access_example(mydataframe):
    a = mydataframe['one']
    b = mydataframe.one
    '''a    1
    b    4
    Name: one, dtype: int32'''
    return a,b