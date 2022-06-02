# basic_python

## Python cheatsheet

## Notes

Basic python

1) Python tutorial (Python 3.3)

2) Check if a particular package is installed

try:

    import numpy

except ImportError:

    print("numpy not installed")

#OR very basic check

import numpy

numpy.__version__

adapted from stackoverflow

3) BioPython for bioinformatics related work

4) BioPython tutorial

5) Parsing a fasta file

########################################################################

# Function - fasta_read3.py

#   Function to parse a fasta file

#

# Dependencies - biopython

#

# Tested on Python 3.3

#

# Usage - python fasta_read3.py List1_sequences.fasta

#

# Acknowledgements - Adapted from

#   1) Response by Zhaorong at

#           http://www.biostars.org/p/710/

#   2) Tutorial at

#           http://www.biopython.org/DIST/docs/tutorial/Tutorial.html#htoc11

#           Section 5.1.1 Reading Sequence Files

#

# Created by Soumya Banerjee

# https://sites.google.com/site/neelsoumya

#

########################################################################

from Bio import SeqIO

def parse_fasta_seq(input_file):

    """Function to read fasta file"""

    for fasta in SeqIO.parse(open(input_file),'fasta'):

        print(fasta.id,fasta.seq.tostring())

if __name__ == "__main__":

    import sys

    parse_fasta_seq(str(sys.argv[1]))

6) Install

 python setup.py install

OR

 sudo python setup.py install

AND

 # install package

 pip install seaborn

AND

# anaconda package manager

conda install fiona

AND

# Install for specific user

pip3 install --user <packagename>

# Install all packages from requirements file

pip3 install -r requirements.txt

OR

pip install -r requirements.txt

# UNINSTALL

pip3 uninstall <name>

7)

        temp_str = '\t'.join(input_file_ptr.readline().split('\t',1)[0:1])

        temp_str = temp_str + '\n'

8) Skip first line/header line of text file while reading text file (adapted from solution in stackoverflow by SilentGhost)

with open('picrust_output_prism_20thjune2014.txt') as f:

      # if you want to ignore first (header) row/line

      next(f)

     

      for x in f:

            # some work in loop

9) Pandas for data manipulation

10) use of join() function

   output_file_healthy_ptr.write('\t'.join([rRow_metadata[6].strip(),rRow_metadata[8].strip(),rRow_metadata[25].strip(),rRow_metadata[27].strip()]) + '\n')

11) convert string to int

int()

12) Read a file line by line and also split a line into columns: use of split()

input_file_risk = 'final_metanalyzed.txt'

with open(input_file_risk, 'r') as input_file_risk_ptr:

       

    reader = csv.reader(input_file_risk_ptr, delimiter = '\t')

       

    for rRow in reader:

            # just rRow will not work with split(); split() needs String

            temp_cols = '\t'.join(rRow).split('\t')

            for lines in temp_cols:

                if lines.strip() == 'stool':

                    output_file_ptr_final.write(str(iInnerCount) + '\n')

ALSO

write to a file and read a file

import csv

input_file_prism = 'processed.cleveland.data'

output_file = 'numbered_cleveland_data.txt'

output_file_ptr  = open(output_file, 'w')

with open(input_file_prism, 'r') as input_file_ptr_prism:

   

    reader = csv.reader(input_file_ptr_prism, delimiter = ',')

    for rRow in reader:

        # convert list to string

        temp_str =  ','.join(rRow)           

        output_file_ptr.write(temp_str + '\n')

output_file_ptr.close()       

if __name__ == "__main__":

    import sys

13) split() function

and then access using array notation [1:] etc

14) strip() function

15) pass (do nothing)

if iLine_number = 1:  

    print('success')

else:

    pass 

16) List Enumeration

    [x[1] for x in pred_prob_array]

    [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

    # Find all index of an array or list with elements greater than > 3

    idx_array = [index for index, x in enumerate(list_x) where x > 3]

17) Pandas

See the 10-minute tutorial and video

18) Pandas

# Great link on pandas tutorial

# Load and read data

import pandas as pd

df_data = pd.read_csv('EdwardsTable2.csv')

print(df_data)

print(df_data.columns) # , header = None

df_data.head()

df_data.loc[:,'Experiment']

df_data.iloc[:,1]

# Experiment is column

#  .loc  with [:, COLUMN_NAME] is like indexing 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pylab as pl

data = pd.read_csv('train.csv')

print(data)

# get columns

data['Activity']

data.describe()

# get whole data in array

data.iloc[0:,0:]

data.hist()

pl.show()

# do database like join operation on two data frames (merge function)

merged_data = pd.merge(trip_data, fare_data)

#print(merged_data)

# head operation in pandas

print(merged_data.head(10))

# output merged file to csv

merged_data.to_csv('innerjoined_data.csv', index=False)

# Alternate style

file_path = os.path.join('data', 'file.csv')

df = pd.read_csv(file_path, header=None)

df.columns = ['name', 'length']

df.length

# Groupby in pandas

df.groupby(['name', 'length'])

# Take mean after doing groupby

df.groupby(['name', 'length']).mean()

df.groupby(['name', 'length']).mean().add_prefix('mean_') # to add prefix mean to every column name (from link)

# Save to csv file

df.to_csv('temp.csv')

# Querying pandas dataframe

# where a and b are columns

df.query('a > b')

df[df.a > df.b] 

# OR use sqldf in pandasql (link)

# pip install -U pandasql

from pandasql import *

temp_df_2 = sqldf("select * "

                      "from df_traffic_times_file"

                      )

# More complex queries in SQL like format

# finding index of rows that match a query

ind = (df_traffic_times_file.from_place == ‘Boston’ )

temp = df_traffic_times_file[ind]

temp_2 = sqldf("select * "

               "from temp where duration_in_traffic > 6200"

               )

print("Most peaky departure time (Boston)\n”,temp_2.departure_time), #temp_2.departure_time.dt.dayofweek)

# Querying pandas data frame without using SQL like syntax

ind = (df_traffic_times_file.to_place == ‘Boston’)

print(df_traffic_times_file[ind].head())

print(df_traffic_times_file[ind].duration_in_traffic)

# Slightly more complex queries

ind = (df_traffic_times_file.to_place == 'Boston') | (df_traffic_times_file.to_place == 'Dortmund' )

print(df_traffic_times_file[ind].head())

# Convert date to pandas datetime

df_traffic_times_file.requested_on = pd.to_datetime(df_traffic_times_file.requested_on)

# Apply function in pandas (using lambda function)

temp_df_mean1 = temp_df.apply(lambda x: np.mean(x) )

# Get day of week

df_traffic_times_file.departure_time.apply( lambda x: x.weekday() )

Other helpful functions in pandas (link)

# fast index to data frame (random access) using set_index()

SUBURB_IMT_DIST_FILE = os.path.join('', 'aggregated_traffic_times.csv')

SUBURB_IMT_DIST_DF = pd.read_csv(SUBURB_IMT_DIST_FILE, header=None)

SUBURB_IMT_DIST_DF.columns = ['terminal', 'suburb', 'mean_id', 'mean_id_stat', 'mean_duration'

                              ,'mean_duration_in_traffic', 'mean_distance'

                              ]

SUBURB_IMT_DIST_DF.set_index(['terminal','suburb'], inplace = True)

SUBURB_IMT_DIST_DF.loc[‘Jamaica Plain’, ‘Port’]

SUBURB_IMT_DIST_DF.loc[‘Jamaica Plain’, ‘Port’].distance

# Replacing strings in pandas dataframe (adapted from stack overflow)

df.from_place.replace({‘Boston Port’ : ‘Boston’}, regex = True, inplace = True)

# where df is the data frame and from_place is the field

# Split pandas data frame based on pattern (from link)

df.from_place = df.from_place.str.split(' Boston', expand=True) # expand = True makes it return a data frame 

 

# Iterate through each row of a dataframe (itertuples iterator)

for temp_frame in df_imt_port_mapping_file.itertuples():

        temp_latlong = str(temp_frame.Lat) + ', ' + str(temp_frame.Long)

# Finding unique values

import pandas as pd

pd.unique(df_traffic_times_file.to_place)

# Plot histogram of series from pandas (link)

pd.Series.hist(temp2.duration_in_traffic)

#Plotting within pandas (plotting from pandas data frame) (courtesy George Mathews)

df_traffic_times_file is a pandas data frame and ind is an index

    ind = (df_traffic_times_file.to_place == 'Boston')

    plt.figure(4)

    df = df_traffic_times_file[ind]

    df = df.set_index(df['departure_time']) #, inplace=True)

    df = df.sort()

    ax = (df['duration_in_traffic']/60.0).plot()#rot = 45)

    ax.set_xlabel("")

    ax.get_xticks() # fill in values on next line

    ax.set_xticks(np.linspace(405000, 405167, 8))

    ax.set_xticklabels(['Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday'], minor=[''], rotation = 45)

    ax.set_ylabel("Duration in traffic (minutes)")

    ax.set_title("Variation")

    plt.tight_layout()

    plt.savefig('timeseries_exploratory_2.png')

19) Data structures in python

20) Anaconda for python

21) How to run in python 2 when you have python 3 installed

conda create -n py2k python=2 anaconda

source activate py2k

22) Checking coding standard and help with refactoring (courtesy Brian Thorne)

        pylint http://www.pylint.org 

23) Python style guide (courtesy Brian Thorne)

Google's Python Style Guide: https://google.github.io/styleguide/pyguide.html

PEP8 style guide: https://www.python.org/dev/peps/pep-0008/

 

24) Good Python IDE (courtesy Brian Thorne)

PyCharm: https://www.jetbrains.com/pycharm/

25) JSON (dumps and loads)

# Serialize 

a = json.dumps({ "heatmap-url": "/results/test.png", "cost": 10 })

# Deserialize

b = json.loads(a)

print(b)

{'cost': 10, 'heatmap-url': '/results/test.png'}

b['cost']

10

b['heatmap-url']

'/results/test.png'

26) Finding the index of an item given a list containing it in Python

 ["foo", "bar", "baz"].index("bar")

1

Finding all index indices of an array with elements greater than 3

np.where(arr_test, > 3)

27) Getting last element of a list

some_list[-1]

28) numpy random number generation

np.random.randn(2, 4)

29) How to frame two for loops in list comprehension python

 [entry for tag in tags for entry in entries if tag in entry]

30) Poisson distribution in numpy

import numpy as np

s = np.random.poisson(5, 10000)

# Display histogram of the sample:

import matplotlib.pyplot as plt

count, bins, ignored = plt.hist(s, 14, normed=True)

plt.show()

np.random.poisson(100000, 10)

31) Element-wise multiplication of two lists

[a*b for a,b in zip(lista,listb)]

32) Print variables (format function)

print("\nRunning: calculation(train_price={},train_freq={}) ...".format(train_price,train_freq))

33) Time function

start = time.clock()

end = time.clock()

print( "optimize() execution time: ", end - start, "seconds" )

34) Interesting data structure

return [

            {'name': ‘Soumya’,

             'fullname': ‘Soumya Banerjee’,

             ‘zip_code': 87106},

            {'name': ‘Banerjee’,

             'fullname': ‘SB’,

             ‘zip_code': 02160}

]

# Array of dict (key-value pairs) (link)

array_dict = {'Name': 'Soumya', 'Age': 17, 'Class': 'First'}, {'Name': 'Sam', 'Age': 23, 'Class': 'First'}

print(array_dict[1])

array_dict[0]['Age']

array_dict[0].keys()

> dict_keys(['Name', 'Class', 'Age'])

array_dict[0].values()

> dict_values(['Soumya', 'First', 17])

array_dict[0].items()

> dict_items([('Name', 'Soumya'), ('Class', 'First'), ('Age', 17)])

35) Use of python docstrings

def get(url, qsargs=None, timeout=5.0):

    """Sends an HTTP GET request.

    :param url: URL for the new request.

    :type url: str

    :param qsargs: Converted to query string arguments.

    :type qsargs: dict

    :param timeout: In seconds.

    :rtype: mymodule.Response

    """

    return request('get', url, qsargs=qsargs, timeout=timeout)

 

36) OR in python if statement

if (d['name'] == name_first_imt) or (d['name'] == name_second_imt):

37) Dict

ret_dict ={}

ret_dict.update({"train_cost": train_cost,

                     "total_teu_volume": float(total_imt_throughput)

        })

OR

dict_mapping_location = \

{  (31, -23): ‘Boston’,

   (30, -62): 'McMurdo’,

}

38) Developer documentation using doxygen

39) Unit testing in python (link and another)

# Doctest: 

def square(x):

    """Squares x.

    >>> square(2)

    4

    >>> square(-2)

    4

    """

    return x * x

if __name__ == '__main__':

    import doctest

    doctest.testmod()

# unittest

import unittest

see link

# Some example code here

from immunemodel import * # import your main modules

import unittest

class BiologicalModelTest(unittest.TestCase):

    @classmethod

    def setUpClass(cls):

cls.i_length_array = 641 # length of features array (number of suburbs)

    def setUp(self):

        """

        Returns:

        """

        print("Loading and initializing ...\n")

        pass

    def tearDown(self):

        """

        Returns:

        """

        print("Clean up after yourself ...\n")

        pass

    def test_map_location_to_name(self):

        """

        Unit testing of map_location_to_name()

        Returns:

        """

self.assertFalse(str_location == ‘Boston1’) 

        self.assertEqual(str_location, ‘Boston’)

        self.assertGreaterEqual(ratio_centroid_googletraffic, 0.9)

        self.assertIsInstance(feature,float) # type check assert

        self.assertNotEqual(str_location, ‘Bosoton1')

if __name__ == "__main__":

    unittest.main()

40) Plotting in python

import matplotlib.pylab as plt

idx_price = train_prices.index(int(train_price))

plt.plot(imt_demands[idx_price], train_price, 'or')

# also index function to find index of an element in an array

# Also code to hold plot and plot labels, titles etc

plt.plot(dist, truck_mean_time, 'or', markersize=15)

plt.hold(True)

plt.plot(dist, imt_mean_time_1, 'ob', markersize-15)

plt.title("Direct Truck vs Intermodal Terminal - Time vs Distance")

plt.xlabel('Distance (meters)' , fontsize=15)

plt.ylabel('Time (hr)' , fontsize=15)

plt.legend(['Direct Truck', 'via IMT'])  #, loc='lower left')

# save figure/plot in python

plt.savefig("hist_pickup_hour.png", dpi=150, alpha=True)

41) word2vec tutorial in kaggle

42) Read from and write to Pandas data frame from kaggle tutorial

import pandas as pd

train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )

for review in train["review"]:

    # do something with review

    sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)

# Write the test results

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

43) Great plots and visualization using plot.ly

44) Socio-economic data from Quandl (link)

import Quandl

data = Quandl.get("FRED/GDP")

data.tail()

45) Metaclass in python (link)

# code courtesy George Mathews

from abc import ABCMeta, abstractmethod

class ImtOpsModeInterface(metaclass=ABCMeta):

    """ abstract class to define an operational mode for an IMT """

    @abstractmethod

    def get_total_ops_cost(self, throughput):

class ImtOpsModeSimple(ImtOpsModeInterface):

46) Numpy range function 

import numpy as np

np.random.uniform(-5, -4.5)

# np.arange

# range

47) Run UNIX command from python

import os

os.system("cut -d ',' -f11-14 innerjoined_data.csv > innerjoined_data_latlong.csv")

# for double quotes within these commands you can use escape (\)

# for example

os.system(" curl -X POST -d '{\"disease\":[\"EFO_0000253\"]}' --header 'Content-Type: application/json' https://platform-api.opentargets.io/v3/platform/public/evidence/filter\?target\=ENSG00000157764  -k > data.json  ")

 

48) Pandas convert from String to datetime (link)

import pandas as pd

# x is a list of strings

pd.to_datetime(pd.Series(x))

# OR

pickup_datetime = pd.to_datetime(pd.Series(merged_data.iloc[:,5]))

# day give day of week (link)

print([x.day for x in pickup_datetime])

# hour gives time of day

print([x.hour for x in pickup_datetime])

49) Pandas datetime operations (link)

50) Concatenate columns (numpy column_stack) (link)

np.column_stack((pickup_day, pickup_hour)))

51) Random forest regressor in python (link)

52) Run UNIX commands from ipython notebook

! module avail 

! ls

53) Plotting histograms in python

import matplotlib.pyplot as plt

plt.figure(1)

plt.hist(pickup_hour, bins = 24)

plt.title("Histogram of pickup times (hour/time of day: 24 hour scale)")

plt.xlabel("pickup times (hour/time of day: 24 hour scale)")

IMG_DPI = 150

plt.savefig("hist_pickup_hour.pdf", dpi=IMG_DPI, alpha=True)

54) Remove all occurrences of an element from a list or array

pickup_longitude = [x for x in pickup_longitude if x != 0]

55) Remove one occurrence of an element from an array

remove()

56) Element wise multiplication of two vectors

# Use zip and list comprehension

[(a - b)**2 for (a,b) in zip(pred_rf_array,test_target_response_fare_amount)]

# Calculate RMSE or SSR

rmse_test = math.sqrt(sum([(a - b)**2 for (a,b) in zip(pred_rf_array,test_target_response_fare_amount)]) /len(pred_rf_array))

57) sqrt (square root) in python

import math

math.sqrt( x )

58) Exponentiation in python

a ** 2

59) Element wise log10 and exponentiation

import numpy as np

target = np.log10(training_target_response_fare_amount),

pred_rf_array = np.power(10, pred_rf_array)

59) Software development tools in python

cookiecutter for package deployment

docker for portable code

60) NLTK (natural language toolkit) in python tutorial (link)

61) Reading and parsing a JSON file (from stackoverflow)

import json

from pprint import pprint

with open('data.json') as data_file:   

    data = json.load(data_file)

pprint(data)

# Accessing fields and elements

data["maps"][0]["id"]

data["masks"]["id"]

data["om_points"]


Using pandas to convert a text to json file (unflatten json)


import pandas as pd

data = pd.read_csv('output_txt.txt', delimiter = '\t', header = None)

# convert to data frame

df = pd.DataFrame(data)

df.to_json('output_json_csv.json')


Using pandas to flatten json file (convert json to text)

data = pd.read_json('manifest.json')

df = pd.DataFrame(data)

df.to_csv('input_json_csv.csv', header = False, index = False)



62) Installing some software or package written in python

pip install biopython

OR

python setup.py build 

python setup.py test 

sudo python setup.py install

63) Designing GUIs in Python (courtesy Simon Luo)

QtDesigner

64) Python library for manipulating datetime

delorean

65) Python library for natural sort

natsort

66) Pandas convert column and entire dataframe to a different type (from stackoverflow)

# to_numeric() function and lambda function

# first column is target

target = data.iloc[0:, 0]

# Convert target to numeric type (one column)

target = pd.to_numeric(target)

# all other columns are training set

train = data.iloc[0:, 1:]

# Convert target to numeric type (all columns)

train = train.apply(lambda x: pd.to_numeric(x) )

67) Create two dimensional array in python numpy

import numpy

numpy.matrix([ [1, 2], [3, 4] ])

68) plotting using seaborn (from link)

tips = sns.load_dataset("tips")

sns.jointplot("total_bill", "tip", tips, kind='reg');

g = sns.pairplot(df, hue='cut')

sns.countplot(x='cut', data=df)

sns.despine()

sns.barplot(x='cut', y='price', data=df)

sns.despine()

g = sns.FacetGrid(df, col='color', hue='color', col_wrap=4)

g.map(sns.regplot, 'carat', 'price')

69) Generic random forests function for regression and classification (on bitbucket)

70) feather -   fast on disk format for data frames in R and python

71) Visualization in python (link) (link to OpenViz conference)

72) Agent based modelling in python (mesa)

73) Serialize object using pickle

pickle.dump()

pickle.load()

pickle.dump(model, open(filename, 'wb'))

pickle.load(open(filename, 'rb'))

74) GUI in Python

Toga (link)

even better package PyWebIO (link)

75) feather -   fast on disk format for data frames in R and python

76) Automatically create new project template with folders etc (link)

cookiecutter https://github.com/drivendata/cookiecutter-data-science

77) NLP in Python

Aylien

monkeylearn

word2vec

lda2vec

78) TPOT python package for automated data science

79) tensorflow for deep learning in python

install using conda environment (link)

conda create -n tensorflow python=3.4

activate tensorflow

# for mac os X Python 3

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-0.8.0rc0-py3-none-any.whl

# At command prompt

python

import tensorflow as tf

80) Code coverage testing using coverage

81) LRU caching of functions (link)

from functools import lru_cache

@lru_cache(maxsize=10000)

def function_name():

82) Working with geometry and shape files

import shapely.geometry

83) Find type of variable or object 

type(list_day_of_week).__name__

84) Use of apply() function (from stackoverflow)

# Convert day of week in numeric to day of week in String

days_of_week = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}

list_day_of_week = list_day_of_week.apply(lambda x: days_of_week[x])

85) Exception handling (try-catch in python) (link)

# try except else

try:

    mean_duration_nominal = \

        SUBURB_IMT_DIST_DF.loc[str_from_location, str_to_location].mean_duration

    except KeyError as key_err:

        print("Name not found in database", key_err)

        return i_exception_time_value, i_exception_time_value

    else:

        return mean_duration_in_traffic, mean_duration_nominal

# raise is like throw

try:

    mean_duration_nominal = \

        SUBURB_IMT_DIST_DF.loc[str_from_location, str_to_location].mean_duration

    except KeyError as key_err:

        raise

    else:

        return mean_duration_in_traffic, mean_duration_nominal

86) C interface or compile python to C using cython and documentation

Cython interface for C GSL (CythonGSL) (link)

87) Parallel programming on multi-cores (joblib)

pip install joblib

AND

multiprocessing

88) Append to list

ratio = []

ratio.append(centroid_distance / new_distance)

89) Inverse lookup on dictionary (find key given value) (from stack overflow)

# .items() function for dict

temp_destination_tuple = [temp_key for temp_key, temp_value in dict_mapping_location.items() if temp_value == ‘Boston’][0]

90) Concatenate strings using + operator

str(temp_frame.Lat) + ', ' + str(temp_frame.Long)

91) Code profile and profiling in Python using cProfile

import cProfile, pstats

cProfile.run('test_model()', 'opt3.p')

stats3 = pstats.Stats('opt3.p').strip_dirs()

stats3.sort_stats('cumulative')

stats3.print_stats(100)

# Also can use time function

start = time.clock()

end = time.clock()

print( "function() execution time: ", end - start, "seconds" )

# Memory profiling and memory leak detection

# Manual garbage collection

from pympler import muppy, summary, tracker

import logging

import gc

memory_profiler

92) Machine learning algorithms book in python

93) Debugger in python (like MATLAB keyboard) (courtesy George Mathews)

import pdb  

pdb.set_trace()

Type c to continue execution (link)

Also use debug command from PyCharm, set breakpoint and you have access to the workspace

94) Static method and class method (link) (courtesy Brian Thorne)

@staticmethod

@classmethod

@classmethod

def setUpClass(cls):

cls.i_length_array = 641 # length of features array (number of suburbs)

# now linked to class and not object

95) Warnings module (courtesy Brian Thorne)

warnings.warn("Using legacy model", DeprecationWarning)

96) Checking where python is installed

which python

# if it needs to be changed then change .bash_profile file 

echo $PATH

vi .bash_profile

97) Manual garbage collection (courtesy Brian Thorne)

import gc

gc.get_stats()

gc.collect()

98) Serialize object

import pickle

pickle.dump()

pickle.load()

99) Libraries to deal with plotting geometry and map objects

fiona # to deal with shape files (fiona.open()  )

descartes (PolygonPatch)

shapely 

Python project

import pyproj

pyproj.Proj()

100) Pretty print (good for printing json objects)

import pprint

101) Randomly shuffle a list

import random

random.shuffle(list)

#shuffles in place

102) Documentation using Sphinx (courtesy Brian Thorne)

see link

and

link (click on source on right hand side)

use makefiles and .rst files

class ImmuneModel():

    """ class to handle truck related methods

     Class level documentation follows

     Notes:

         1. 

         2. 

     .. rubric:: Footnotes

     ..  [1] Technically

    Args:

        None:

    Attributes:

        

    Raises:

    For example::

            enc = public_key.encrypt(1337)

    """

103) Refactoring code in PyCharm (link) (courtesy George Mathews)

104) Fast alternative implementation of python (PyPy) (compiled)

 

105) Genomic data manipulation and visualization in python (link)

106) Choose from a list or array randomly with some probability using numpy.random.choice

import numpy as np

str_single_aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K',

                 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

str_TCR_sequence = np.random.choice(str_single_aa, self.i_length_TCR) #, p=[0.5, 0.1, 0.1, 0.3])

107) Get value of attribute or member of a class

# receptor is name of member

epithelialcell.__getattribute__("receptor")

108) Assign to a String in python (link)

Strings are immutable in Python

str_TCR_sequence = ''

for aa in list:

        str_TCR_sequence = str_TCR_sequence + aa

109) Parse or get command line arguments in main

if __name__ == "__main__":

    import sys

    func_column_parser(str(sys.argv[1]),str(sys.argv[2]))

110) Writing or saving a python list to a file (link)

    with open("file.csv", 'w', newline='\n') as outfile:

        wr = csv.writer(outfile)#, quoting=csv.QUOTE_ALL)

        for item in list_all_escape:

            wr.writerow(item)

OR

f=open('file_ALL_peptides.csv','w')

for item in list_giant_ALL_COMBINED_peptides:

       f.writelines(item)

       f.writelines("\n")

f.close()

111) Element-wise division of two arrays

numpy.divide

112) Flattening a list of lists (link)

 [x for sublist in list_ALL_peptides_data  for x in sublist]

113) Adding multiple figures to a plot (using plt.hold(True) and plt.show()  )

import matplotlib.pyplot as plt

plt.plot(array_interaction_intervals, list_tcr_escapecount_AUTO, '.b')

plt.hold(True)

# ..... more code here

plt.plot(array_interaction_intervals, list_tcr_escapecount_AUTO, '.r')

plt.show()

114) Get number of lines in a file in python (link)

def file_number_of_lines(fname):

    """

    Adapted from http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python

    Args:

        fname:

    Returns:

    """

    with open(fname) as f:

        for i, l in enumerate(f):

            pass

    return (i + 1)

115) Divide two lists element-wise (link)

from operator import truediv

list_per_escaping_auto_overALLescaping = map(truediv, list_tcr_escapecount_AUTO, list_tcr_escapecount_ALL)

[a for a in list_per_escaping_auto_overALLescaping]

116) Lean and efficient MicroPython

117) enumerate (get index of element and element in list comprehension) (link)

118) Filter out some elements from list (filter) (stackoverflow)

119) Set or unique list (stackoverflow)

import collections

d = collections.defaultdict(set)

d[1].add(<element>)

120) Change working directory from python

DATA_DIR = "data" 

os.chdir( os.path.join(DATA_DIR) )

121) replace string in python

str.replace()

"er.py".replace(".", "")

122) GUI in python using tkinter (example) (link) (examples)

from tkinter import *

# create a window

window = Tk()

window.geometry("312x324")

window.title("Conversational AI")

window.mainloop()

123) turtle in python (link)

import turtle

wn = turtle.Screen()

alex = turtle.Turtle()

alex.forward(160)

alex.left(90)

alex.forward(89)

wn.bgcolor("lightgreen")        # set the window background color 

alex.color("blue")

124) barplot in python (link)

import matplotlib.pyplot as plt

x  = [a for a in range(1,20 + 1)] # 20 aa

y  = [b for (a,b) in list_eachaa_number_matched] # number of times each aa occurs

str_labels = [a for (a,b) in list_eachaa_number_matched] # get names of each aa

width = 1/1.5

plt.bar(x, y, width, color="blue")

plt.xlabel(str_labels)

plt.ylabel("Frequency of occurrence of each amino acid")

plt.show()

plt.savefig("analyze_allpeptides_GIANTCELL_reactagainst_autoBUTESCAPE.eps")

125) Read data from websites, parse and put into pandas (link)

126) Use BeautifulSoup to parse HTML and tables (link)

127) matplotlib plot symbols

b     blue          .     point              -     solid

           g     green         o     circle             :     dotted

           r     red           x     x-mark             -.    dashdot 

           c     cyan          +     plus               --    dashed   

           m     magenta       *     star             (none)  no line

           y     yellow        s     square

           k     black         d     diamond

w     white         v     triangle (down)

                               ^     triangle (up)

                               <     triangle (left)

                               >     triangle (right)

                               p     pentagram

                               h     hexagram

import matplotlib.pyplot as plt

str_color_plot = 'dg'

plt.plot(array_interaction_intervals, list_tcr_escapecount_AUTO, str_color_plot, markersize=15)

plt.title("Percentage")

plt.xlabel("Number", fontsize = 15)

plt.ylabel("Percentage", fontsize = 15)

 

128) matplotlib resource and tutorial VERY GOOD (link)

         matplotlib ggplot style plots (link)

        

    from matplotlib import pyplot as plt     plt.style.use('ggplot')

         other styles

         plt.styles.available

129) Remove all occurrences of an element from a list (adapted from stackoverflow)

use of filter

list( filter( lambda a: a!=0, list_degeneracy_against_nonself ) )

130) Shift an array (link)

from collections import deque

items = deque( [ 'L', 'F', 'L', 'F' ] )

items.rotate(1)  # becomes F, L, F, L

131) remove duplicates from list (from stackoverflow)

t = list(set(t))

132) filename manipulation

using endswith, startswith and glob

for temp_file in os.listdir(DATA_DIR):

    #for temp_file in glob.glob('*.csv'):

        if temp_file.startswith("str_array__" + "0_"):

        #if temp_file.endswith(".csv"):

132) Python virtual environments (courtesy Joe Kearney) (link)

python3 -m venv ~/.venvs/venv_name  # create the venv

source ~/.venvs/venv_name/bin/activate # to enter virtualenv

deactivate # to leave

133) scipy datetime functions (link)

134) Check dimension of numpy array

array.ndim

135) ** to call function (link)

def foo(x,y,z):     print("x=" + str(x))     print("y=" + str(y))     print("z=" + str(z))

mydict = {'x':1,'y':2,'z':3}foo(**mydict)

136) .get() method to get a value for a key in a Dict (link)

dict = {'Name': 'Zabra', 'Age': 7}

dict.get('Age')

137) matplotlib style or backend like ggplot

import matplotlib.pyplot as plt

plt.style.use('ggplot')

138) Installing python packages using pip on Windows

python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

139) Running a Python program on Windows

py deep_leaarning_keras_uci.py

140) Shape of numpy array

np.shape(x_train)

141) Plot images like images of numbers (link)

from keras.datasets import mnist

# Load data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# rescale to be between 0 and 1

x_train = x_train.astype('float32')/255

y_train = y_train.astype('float32')/255

x_test  = x_test.astype('float32')/255

y_test  = y_test.astype('float32')/255

x_train.shape

x_train.shape[1:] # 28 x 28

np.prod(x_train.shape[1:])

# flatten to vector for each image (60000 x 784)

tpl_flatten_new_dimensions = (  len(x_train),  np.prod(x_train.shape[1:])  )

x_train = np.reshape( x_train,  tpl_flatten_new_dimensions )

tpl_flatten_new_dimensions = (  len(x_test),  np.prod(x_test.shape[1:])  )

x_test = np.reshape( x_test,  tpl_flatten_new_dimensions )

######################################

# Plotting code

######################################

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 4))

plt.imshow(x_test[10].reshape( (28,28) ))

plt.show()

142) In NLP, replace a list of stopwords with blanks or " " (link)

# https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Exercise%20-%20Answer.ipynb

sentences = []

labels    = []

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",

             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",

             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",

             "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",

             "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",

             "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",

             "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",

             "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",

             "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",

             "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",

             "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",

             "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",

             "yourself", "yourselves" ]

# open file and read and do data munging

with open('bbc-text.csv', 'r') as csvfile:

    reader = csv.reader(csvfile, delimiter=',')

    next(reader)

    # for every row

    for row in reader:

        labels.append(row[0])

        sentence = row[1]

        # in this sentence, replace stopwords with ''

        for word in stopwords:

            sentence.replace(" " + word + " ", " ")

        # append this sentence to sentences

        sentences.append(sentence)

    # next row

143) Operations with dict  like building a reverse dictionary and list and getting item in a dict

for list

.items()

gets all items

for dict

.get()

gets value of key

reverse_word_index = dict( [(value,key) for (key,value) in word_index.items() ] )

def decode_sentence(text):

    return (  ' '.join( reverse_word_index.get(i, '?') for i in text  ) )

dict = {'Name': 'Zabra', 'Age': 7}

print "Value : %s" %  dict.get('Age')

144) Saving a numpy array/matrix to disk

np.savetxt('x_test.txt', x_test, delimiter = ',') 

145) Creating a numpy array of zeros using np.zeros (link)

# get shape or dimensions or dim of numpy array using np.shape

(i_num_patients_testset, i_num_columns_categorical_testset) = np.shape(x_test_orig[:,i_categorical_offset:])

        

df_class_contrastive_ml = np.zeros((i_num_patients_testset, i_num_columns_categorical_testset))

146) Combinations of different numbers (link)

list_temp = [1, 2, 3, 4]

combinations(list_temp  , 2)

tuple_temp = [x for x in combinations(list_temp, 2) ]

147) Distribution in seaborn histogram and create good histograms in python

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

# this helps create good plots

sns.set()

 

############################

# generate random numbers

############################

list_numbers = np.random.rand(10000)

############################

# Plot distribution

############################

plt.figure()

plt.hist(list_numbers)

plt.show()

sns.distplot(list_numbers)

# even better seaborn sns distplot

sns.distplot(list_numbers, kde=True, color='darkblue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth':4})

# plot exponential distribution

sns.distplot(np.random.exponential(1,100))

# also plot with vertical lines

lb = np.percentile(list_normal_numbers, 2.5)

ub = np.percentile(list_normal_numbers, 97.5)

sns.distplot(list_normal_numbers)   

plt.vlines(lb, 0, 1)

plt.vlines(ub, 0, 1)

148) Help

help(sns.set)

149) Linear regression in python

A = 4

k = 2.7

i_data_points = 400

x = np.random.rand(i_data_points)

y = np.exp(-k*x)

plt.figure()

plt.plot(y, x, '.')

plt.show()

# now take log

log_y = np.log(y)

###############################

# Now do a linear regression

################################

reg = np.polyfit(x, log_y, 1)

fit = reg[1] + reg[0]*x

plt.figure()

plt.plot(fit, log_y, '.b')

plt.show()

########################################

# Bootstrap to get confidence intervals

########################################

# now repeat this process many times independently

# boostrap rounds

# list to store parameters

list_parameters = []

for _ in np.arange(0, 1000):

   

    # do this 1000 times

   

    bootstrpa_indices = np.random.randint(0, i_data_points, i_data_points)

    # now you have indices

    # get or draw those from the original data

    boot_log_y = log_y[bootstrpa_indices]

    boot_x     = x[bootstrpa_indices]

    reg_boot = np.polyfit(boot_x, boot_log_y, 1)

    print(reg_boot[0])

    print(reg_boot[1])

    # append estiomate of paremetr to list

    list_parameters.append(reg_boot[0])

lb = np.percentile(list_normal_numbers, 2.5)

ub = np.percentile(list_normal_numbers, 97.5)

sns.distplot(list_normal_numbers)   

plt.vlines(lb, 0, 1)

plt.vlines(ub, 0, 1)

###################################

# Another easy bootstrap example

###################################

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

sns.set()

list_normal_numbers = np.random.randn(1000) # standard normal N(0,1)

# now plot distribition of these bootstrapped estimates

sns.distplot(list_normal_numbers)   

np.percentile(list_normal_numbers, 2.5)

np.percentile(list_normal_numbers, 97.5)

lb = np.percentile(list_normal_numbers, 2.5)

ub = np.percentile(list_normal_numbers, 97.5)

sns.distplot(list_normal_numbers)   

plt.vlines(lb, 0, 1)

plt.vlines(ub, 0, 1)

150) Seaborn plotting with vertical lines 

lb = np.percentile(list_normal_numbers, 2.5)

ub = np.percentile(list_normal_numbers, 97.5)

sns.distplot(list_normal_numbers)   

plt.vlines(lb, 0, 1)

plt.vlines(ub, 0, 1)

151) scipy stats functions

import scipy.stats as st

st.norm.ppf(0.025)

st.norm.ppf(0.975)

152) Numpy sort

import numpy as np

np.sort(a, axis = 1) # axis  = 0 is rows and axis = 1 is columns

153) numpy sign function

np.sign(-10)

154) Sample from uniform distribution

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as st

from scipy.stats import uniform, norm

import pandas as pd

sns.set()

#fig, axes = plt.subplots()

#axes.plot()

x = np.random.rand(100)

y = 1 + x + st.uniform.rvs(-0.5, 0.5)

sns.distplot(y)

sns.distplot(x)

plt.figure()

plt.plot(y, x, '.r')

plt.show()

155) In pandas, split into test and training using pandas.dataframe.sample

import pandas as pd

data =  pd.read_csv('/Users/soumya/Documents/abalone.data', header = None)

# split into test and training

data.sample(frac=0.7)

156) set() set in python

set([1,2,2])

{1,2}

157) linear spaced array using linspace

import numpy as np

import seaborn as sns

x = np.linspace(0, 10, 200)

sns.distplot(x)

158) t distribution in python using scipy st.t.pdf()

import numpy as np

import seaborn as sns

import scipy.stats as st

x = np.linspace(0, 10, 200)

#sns.distplot(x)

t_distibution = st.t.pdf(x, df = 2)

sns.distplot(t_distibution)

159) Calculate covariance using np.cov

import numpy as np

np.cov( df_data.iloc[:,1], df_data.iloc[:,0] )

160) Scatter plot

import matplotlib.pyplot as plt

plt.scatter(x, y)

plt.show()

161) LASSO penalised regression in python using the Lasso package in sklearn (code from bootcamp private)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso

sns.set()

# set seed

np.random.seed(10)

NSamps   = 100

Ncov     = 200

Ntruecov = 10

beta = np.zeros((Ncov))

trueIdx = np.random.choice( np.arange(Ncov), replace = False, size=Ntruecov )

beta[trueIdx] = np.random.rand(Ntruecov) * 2 + 0.1

beta[trueIdx] *= np.round( np.random.rand(Ntruecov) ) * 2 - 1

beta = beta.reshape(-1, 1)

noiseFloor = 10

XRange = 20

X = np.random.rand(NSamps, Ncov) * XRange  - (XRange/2.0)

Y = np.dot(X, beta) + np.random.rand(NSamps, 1) * noiseFloor

print(np.shape(X))

print(np.shape(Y))

plt.figure()

plt.plot(Y, X, '-b')

plt.show()

sns.distplot(Y)

###################

# Fit LASSO model

###################

lasso_object = Lasso(alpha = 1)

lasso_object.fit(X, Y)

coef = lasso_object.coef_

print(coef)

plt.figure()

plt.plot(coef)

plt.show()

# and OLS model set alpha = 0

ols_object = Lasso(alpha = 1)

ols_object.fit(X, Y)

coef_ols = ols_object.coef_

print(coef_ols)

plt.figure()

plt.plot(coef_ols)

plt.show()

# how many coefficient set to 0?

len(np.where(coef == 0))

len(np.where(coef_ols == 0))

lasso_object.predict(X)

162) PCA in python (data science bootcamp link)

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

from sklearn.decomposition import PCA

data_X = np.loadtxt('mnist2500_X.txt')

data_Y = np.loadtxt('mnist2500_labels.txt')

sns.distplot(data_Y)

imgplot = 0

data_plot = data_X[imgplot]

data_plot_reshape = data_plot.reshape(28,28) # since this in in 784 size so make it 28 by 28

plt.imshow(data_plot_reshape)

data_normalised = data_X - data_X.mean(axis = 0) # normalise data

pca_object = PCA(n_components = data_X.shape[1])

pca_object.fit(data_normalised)

data_normalised.shape[1]

# get pricnipal components

pricn_components = pca_object.components_

pricn_components[0]

# Plot first principal component

plt.imshow(pricn_components[0].reshape(28,28))

projection = np.dot(data_X, pricn_components[0])

# use PC1 to project data on to 

plt.imshow(projection.reshape(50,50))

plt.scatter(pricn_components[0], pricn_components[1] , alpha = 0.7 )
Page updated
Google Sites
Report abuse
