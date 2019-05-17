#import os
#os.chdir(os.getcwd())
"""try:
    import pandas as pd
    import numpy as np
except ImportError  as error:
    print("Pandas or Numpy is not available in this System, please install and use the tool")



try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.externals import joblib
except ImportError  as error:
    print("Sklearn and Subpackage of Sklearn is not available in this System.")

try:
    import datetime
    import dateutil.relativedelta

except ImportError as error:
    print("datetime or dateutil package is not available")

try:
    from xgboost import XGBClassifier
except ImportError as error:
    print("XGBOOST PAckage is not found in this machine")

try:
    from scipy.spatial.distance import cosine
    from sklearn.metrics.pairwise import euclidean_distances
except ImportError as error:
    print("scipy.spatial.distance import cosine or from sklearn.metrics.pairwise import euclidean_distances is not found in this machine" )
    
try:
    import os
    import math

except ImportError as error:
    print("OS , MATH packages are not found in this machine")


try:
    from itertools import chain, product
    import pandas.core.algorithms as algos
    import scipy.stats.stats as stats
    import statsmodels.api as sm
    import collections
    import matplotlib
    matplotlib.use("agg") 
    import matplotlib.pyplot as plt
    from matplotlib import rcParams    
except ImportError as error:
    print("one or more than one packages are missing from the below package list : \n itertools \n pandas.core.algorithms \n scipy.stats.stats  \n statsmodels.api \n collection \n matplotlib \n matplotlib.pyplot \n from matplotlib import rcParams")


try:
    from MVP.FeatureEngineering import FeatureEngineering
except:
    print("Class FeatureEngineering not found")

try:    
    from MVP.XgbSelection import XgbSelection
except:
    print("Class XgbSelection not found")
    

try:
    from  MVP.dataDictionary import myModel
except:
    print("Class myModel is not found")

try:
    from MVP.Imputer import Imputer
except:
    print("Class Imputer is Not found")

try:
    from MVP.OutlierTreatment import OutlierTreatment
except:
    print("OutlierTreatment Class is not found")
    
try:
    from MVP.CreateDummies import CreateDummies
except:
    print("Class CreateDummies not available for Import")
"""
