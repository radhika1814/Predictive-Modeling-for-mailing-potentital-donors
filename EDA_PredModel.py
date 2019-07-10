
# coding: utf-8

# # Solution and code for question 1 and 2

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pandas import DataFrame
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv('C:/Users/HP/Desktop/SampleDonorData.csv')


# In[3]:


dataset.hist(column='DONOR_AGE')


# In[4]:


datasetcont=dataset.drop(dataset.columns[[6,8,9,18]], axis=1)
#'URBANICITY','HOME_OWNER','DONOR_GENDER','recency_freq_status'


# We can see different summary statistics the mean, standard deviation, median would be the 50% value

# Summary statistics for all the variables in the data. 1b.

# In[5]:


dataset.describe()


# Correlation plot for continour variables

# In[6]:


import seaborn as sns
corr = datasetcont.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

