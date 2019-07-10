
# coding: utf-8

# # Q5) Build the best classification model using machine learning models

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
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Reading our data into dataset 
dataset = pd.read_csv('C:/Users/HP/Desktop/SampleDonorData.csv')


# In[3]:


#Dealing with null and missing values by using method =fill
dataset.isnull().any()
dataset = dataset.fillna(method='ffill')
dataset.dtypes


# In[4]:


#Counting categorical variable values to normalize the data
dataset["URBANICITY"].value_counts()  
dataset["HOME_OWNER"].value_counts()
dataset["DONOR_GENDER"].value_counts()
dataset["recency_freq_status"].value_counts()


# In[5]:


#Replacing categorical values with numbers
cleanup_nums = {"URBANICITY":     {"C": 0, "R": 1,"S":2,"T":3, "U":4,"?":5},
                "HOME_OWNER": {"H": 1, "U": 0},
               "DONOR_GENDER": {"M":1, "F":2,"U":3,"A":4},
                "recency_freq_status":{"A1":0,"A2":1,"A3":2,"F1":3,"S2":4,"S4":5,"S3":6,"A4":7,"N1":8,"N2":9,"N3":10,"E1":11,"N4":12,"L1":13,"S1":14,"E4":15,"F2":16,"E2":17,"E3":18,"F4":19,"F3":20,"L2":21,"L4":22}
               }


# In[6]:


dataset.replace(cleanup_nums, inplace=True)
dataset.head()


# In[7]:


dataset.dtypes


# In[8]:


#For classification our target variable is TARGET_B
X = dataset[['MONTHS_SINCE_ORIGIN', 'DONOR_AGE', 'IN_HOUSE', 'CLUSTER_CODE', 'INCOME_GROUP', 'PUBLISHED_PHONE', 'WEALTH_RATING','MEDIAN_HOME_VALUE','MEDIAN_HOUSEHOLD_INCOME','PCT_OWNER_OCCUPIED','PEP_STAR','RECENT_STAR_STATUS','RECENT_CARD_RESPONSE_PROP','MONTHS_SINCE_LAST_PROM_RESP','LAST_GIFT_AMT','NUMBER_PROM_12','MONTHS_SINCE_LAST_GIFT','URBANICITY','HOME_OWNER','DONOR_GENDER','recency_freq_status']].values
y= dataset['TARGET_B'].values


# To get an idea of our data to see average number of donors and non-donors. We can see that we have maximum number of non-donors.

# In[9]:


plt.figure(figsize=(15,10))
plt.tight_layout()
seabornInstance.distplot(dataset['TARGET_B'])


# Splitting the data into test and train with a proportion of 70-30 split and shuffle=True.

# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, shuffle=True)


# Feature scaling for better accuarcy of models

# In[11]:


from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
# apply same transformation to test data
X_test = scaler.transform(X_test)  


# # Model1: Random Forest Classification

# In[12]:


from sklearn.ensemble import RandomForestClassifier
#Setting the tuning parameter for the classifier for better accuracy and precision.
regressor = RandomForestClassifier(n_estimators=500, random_state=0)  
regressor.fit(X_train, y_train)  
y_pred = regressor.predict(X_test)  


# In[13]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1


# In[14]:


#Calculating the Accuracy, Precision and Recall for the model.
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred))


# # Model2: Naive Bayes Classification

# In[15]:


from sklearn.naive_bayes import GaussianNB
modelnb = GaussianNB()


# In[16]:


# fitting x samples and y classes 
modelnb.fit(X_train, y_train) 
y_prednb = modelnb.predict(X_test)  


# In[17]:


df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_prednb})
df3 = df2.head(25)
df3


# In[18]:


print(confusion_matrix(y_test,y_prednb))  
print(classification_report(y_test,y_prednb))  
print(accuracy_score(y_test, y_prednb))  


# # Model3: KNN classifier

# In[19]:


from sklearn.neighbors import KNeighborsClassifier

modelknn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
modelknn.fit(X_train,y_train)


# In[20]:


y_predknn = modelknn.predict(X_test)  


# In[21]:


df4 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predknn})
df5 = df4.head(25)
df5


# In[22]:


print(confusion_matrix(y_test,y_predknn))  
print(classification_report(y_test,y_predknn))  
print(accuracy_score(y_test, y_predknn)) 


# # Model 4: Support Vector Machine Classifier

# In[23]:


# import support vector classifier 
from sklearn import svm # "Support Vector Classifier" 
modelsvm = svm.SVC() 
modelsvm.fit(X_train, y_train)


# In[24]:


y_predictsvm = modelsvm.predict(X_test)


# In[25]:


df6 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predictsvm})
df7 = df6.head(25)
df7


# In[26]:


print(confusion_matrix(y_test,y_predictsvm))  
print(classification_report(y_test,y_predictsvm))  
print(accuracy_score(y_test, y_predictsvm))


# # Model 5: Decision Tree Classifier

# In[27]:


from sklearn import tree

modeldt = tree.DecisionTreeClassifier()
modeldt.fit(X_train, y_train)


# In[28]:


y_preddt= modeldt.predict(X_test)


# In[29]:


df8 = pd.DataFrame({'Actual': y_test, 'Predicted': y_preddt})
df9 = df8.head(25)
df9


# In[30]:


print(confusion_matrix(y_test,y_preddt))  
print(classification_report(y_test,y_preddt))  
print(accuracy_score(y_test, y_preddt))


# # Model 6: Multilayer Perceptron classifier

# In[31]:


from sklearn.neural_network import MLPClassifier

modelmlp = MLPClassifier(shuffle=False)

modelmlp.fit(X_train, y_train)                         


# In[32]:


y_predmlp= modelmlp.predict(X_test)


# In[33]:


df10 = pd.DataFrame({'Actual': y_test, 'Predicted': y_predmlp})
df11 = df10.head(25)
df11


# In[34]:


print(confusion_matrix(y_test,y_predmlp))  
print(classification_report(y_test,y_predmlp))  
print(accuracy_score(y_test, y_predmlp))

