
# coding: utf-8

# # Regression Q3 and Q4building linear regression model and comparing with other regression models 

# In[1]:


#Importing the required packages
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from pandas import DataFrame
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading the csv file
dataset = pd.read_csv('C:/Users/HP/Desktop/SampleDonorData.csv')


# In[3]:


#This shows the number of rows and columns in our data
dataset.shape


# In[4]:


#Summary statistics of the data
dataset.describe()


# In[5]:


dataset.isnull().any() #Checking for null, Na's in our data
#Filling the null/NA values with Fill method
dataset = dataset.fillna(method='ffill')
dataset.dtypes #Checking the datatypes of the variables



# In[6]:


#Counting values for all the categorical variables in the data so we know how to convert and normalize these values to run in our ML models
dataset["URBANICITY"].value_counts()  
dataset["HOME_OWNER"].value_counts()
dataset["DONOR_GENDER"].value_counts()
dataset["recency_freq_status"].value_counts()


# In[7]:


#Replacing the categorical values with numerical values
cleanup_nums = {"URBANICITY":     {"C": 0, "R": 1,"S":2,"T":3, "U":4,"?":5},
                "HOME_OWNER": {"H": 1, "U": 0},
               "DONOR_GENDER": {"M":1, "F":2,"U":3,"A":4},
                "recency_freq_status":{"A1":0,"A2":1,"A3":2,"F1":3,"S2":4,"S4":5,"S3":6,"A4":7,"N1":8,"N2":9,"N3":10,"E1":11,"N4":12,"L1":13,"S1":14,"E4":15,"F2":16,"E2":17,"E3":18,"F4":19,"F3":20,"L2":21,"L4":22}
               }


# In[8]:


#Replacing the categorical values of the 4 variables with the above values.
dataset.replace(cleanup_nums, inplace=True)
dataset.head()


# In[9]:


dataset.dtypes #Checking the datatypes after replacing (Urbanicity dtype changed from object to int64)


# In[10]:


#Storing the target variable in y and the features in x, here we do not consider the variables that were highly correlated thus, when using TARGET_D as target variable y we do not use TARGET_B in features(X) and Months_Since_Origin is highly correlated with  Month_Since_First_Gift so we consider only one of those features.
X = dataset[['MONTHS_SINCE_ORIGIN', 'DONOR_AGE', 'IN_HOUSE', 'CLUSTER_CODE', 'INCOME_GROUP', 'PUBLISHED_PHONE', 'WEALTH_RATING','MEDIAN_HOME_VALUE','MEDIAN_HOUSEHOLD_INCOME','PCT_OWNER_OCCUPIED','PEP_STAR','RECENT_STAR_STATUS','RECENT_CARD_RESPONSE_PROP','MONTHS_SINCE_LAST_PROM_RESP','LAST_GIFT_AMT','NUMBER_PROM_12','MONTHS_SINCE_LAST_GIFT','URBANICITY','HOME_OWNER','DONOR_GENDER','recency_freq_status']].values
y= dataset['TARGET_D'].values


# In[11]:


#To check the average amount of donations maximum donations is 0 and there are significant donations in the amount of 0-25
plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['TARGET_D'])


# In[12]:


#To divide the data into train and test data, here I have taken a split of 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle = True)


# # Model 1: Linear Regression 

# In[13]:


#Fitting the model and training it on the train data
regressor = LinearRegression(normalize=True)  
regressor.fit(X_train, y_train)


# In[14]:


#Feature prediction
predictions = regressor.predict(X_train)


# In[15]:


y_pred = regressor.predict(X_test)


# In[16]:


#To compare the model prediction with the actual values for donation amount(Target_d)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1


# To see the predicted values and the actual values for comparison

# In[17]:


df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[18]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# # Model 2: Random Forest Regressor

# In[19]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)


# In[20]:


# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Print out the errors
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[21]:


df2 = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
df3 = df2.head(25)
df3


# # Model 3: Decision Tree Regressor

# In[22]:


# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)  
  
# fit the regressor with X and Y data 
regressor.fit(X_train, y_train) 


# In[23]:


predictionsdtr = regressor.predict(X_test)

df4 = pd.DataFrame({'Actual': y_test, 'Predicted': predictionsdtr})
df5 = df4.head(25)
df5


# In[26]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictionsdtr))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictionsdtr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictionsdtr)))


# # Model 4: Support Vector Machine Regressor

# In[28]:


from sklearn.svm import SVR 
model = SVR()


# In[29]:


model.fit(X_train, y_train)


# In[30]:


predictionssvr = model.predict(X_test)

df6 = pd.DataFrame({'Actual': y_test, 'Predicted': predictionssvr})
df7 = df6.head(25)
df7


# In[31]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictionssvr))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictionssvr))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictionssvr)))

