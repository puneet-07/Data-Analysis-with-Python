#!/usr/bin/env python
# coding: utf-8

#  <a href="https://www.bigdatauniversity.com"><img src = "https://ibm.box.com/shared/static/ugcqz6ohbvff804xp84y4kqnvvk3bq1g.png" width = 300, align = "center"></a>
# 
# <h1 align=center><font size = 5>Data Analysis with Python</font></h1>

# # House Sales in King County, USA

# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

# <b>id</b> :a notation for a house
# 
# <b> date</b>: Date house was sold
# 
# 
# <b>price</b>: Price is prediction target
# 
# 
# <b>bedrooms</b>: Number of Bedrooms/House
# 
# 
# <b>bathrooms</b>: Number of bathrooms/bedrooms
# 
# <b>sqft_living</b>: square footage of the home
# 
# <b>sqft_lot</b>: square footage of the lot
# 
# 
# <b>floors</b> :Total floors (levels) in house
# 
# 
# <b>waterfront</b> :House which has a view to a waterfront
# 
# 
# <b>view</b>: Has been viewed
# 
# 
# <b>condition</b> :How good the condition is  Overall
# 
# <b>grade</b>: overall grade given to the housing unit, based on King County grading system
# 
# 
# <b>sqft_above</b> :square footage of house apart from basement
# 
# 
# <b>sqft_basement</b>: square footage of the basement
# 
# <b>yr_built</b> :Built Year
# 
# 
# <b>yr_renovated</b> :Year when house was renovated
# 
# <b>zipcode</b>:zip code
# 
# 
# <b>lat</b>: Latitude coordinate
# 
# <b>long</b>: Longitude coordinate
# 
# <b>sqft_living15</b> :Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# 
# 
# <b>sqft_lot15</b> :lotSize area in 2015(implies-- some renovations)

# You will require the following libraries 

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1.0 Importing the Data 

#  Load the csv:  

# In[2]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# 
# we use the method <code>head</code> to display the first 5 columns of the dataframe.

# In[3]:


df.head()


# #### Question 1 
# Display the data types of each column using the attribute dtype, then take a screenshot and submit it, include your code in the image. 

# In[4]:


df.dtypes


# We use the method describe to obtain a statistical summary of the dataframe.

# In[5]:


df.describe()


# # 2.0 Data Wrangling

# #### Question 2 
# Drop the columns <code>"id"</code>  and <code>"Unnamed: 0"</code> from axis 1 using the method <code>drop()</code>, then use the method <code>describe()</code> to obtain a statistical summary of the data. Take a screenshot and submit it, make sure the inplace parameter is set to <code>True</code>

# In[6]:


df.drop("id", axis = 1, inplace = True)
df.drop("Unnamed: 0", axis = 1, inplace = True)

df.describe()


# we can see we have missing values for the columns <code> bedrooms</code>  and <code> bathrooms </code>

# In[7]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# 
# We can replace the missing values of the column <code>'bedrooms'</code> with the mean of the column  <code>'bedrooms' </code> using the method replace. Don't forget to set the <code>inplace</code> parameter top <code>True</code>

# In[8]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# 
# We also replace the missing values of the column <code>'bathrooms'</code> with the mean of the column  <code>'bedrooms' </codse> using the method replace.Don't forget to set the <code> inplace </code>  parameter top <code> Ture </code>

# In[9]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[10]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# # 3.0 Exploratory data analysis

# #### Question 3
# Use the method value_counts to count the number of houses with unique floor values, use the method .to_frame() to convert it to a dataframe.
# 

# In[11]:


df['floors'].value_counts().to_frame()


# ### Question 4
# Use the function <code>boxplot</code> in the seaborn library  to  determine whether houses with a waterfront view or without a waterfront view have more price outliers .

# In[12]:


sns.boxplot(x="waterfront", y="price", data=df)


# ### Question 5
# Use the function <code> regplot</code>  in the seaborn library  to  determine if the feature <code>sqft_above</code> is negatively or positively correlated with price.

# In[13]:


sns.regplot(x="sqft_above", y="price", data=df, ci = None)


# 
# We can use the Pandas method <code>corr()</code>  to find the feature other than price that is most correlated with price.

# In[14]:


df.corr()['price'].sort_values()


# # Module 4: Model Development

# Import libraries 

# In[15]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 
# We can Fit a linear regression model using the  longitude feature <code> 'long'</code> and  caculate the R^2.

# In[16]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# ### Question  6
# Fit a linear regression model to predict the <code>'price'</code> using the feature 'sqft_living' then calculate the R^2. Take a screenshot of your code and the value of the R^2.

# In[17]:


X1 = df[['sqft_living']]
Y1 = df['price']
lm = LinearRegression()
lm
lm.fit(X1,Y1)
lm.score(X1, Y1)


# ### Question 7
# Fit a linear regression model to predict the 'price' using the list of features:

# In[19]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# the calculate the R^2. Take a screenshot of your code

# In[20]:


X2 = df[features]
Y2 = df['price']
lm.fit(X2,Y2)
lm.score(X2,Y2)


# #### this will help with Question 8
# 
# Create a list of tuples, the first element in the tuple contains the name of the estimator:
# 
# <code>'scale'</code>
# 
# <code>'polynomial'</code>
# 
# <code>'model'</code>
# 
# The second element in the tuple  contains the model constructor 
# 
# <code>StandardScaler()</code>
# 
# <code>PolynomialFeatures(include_bias=False)</code>
# 
# <code>LinearRegression()</code>
# 

# In[21]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# ### Question 8
# Use the list to create a pipeline object,  predict the 'price', fit the object using the features in the list <code> features </code>, then fit the model and calculate the R^2

# In[22]:


pipe=Pipeline(Input)
pipe


# In[23]:


pipe.fit(X,Y)


# In[24]:


pipe.score(X,Y)


# # Module 5: MODEL EVALUATION AND REFINEMENT

# import the necessary modules  

# In[25]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# we will split the data into training and testing set

# In[26]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# ### Question 9
# Create and fit a Ridge regression object using the training data, setting the regularization parameter to 0.1 and calculate the R^2 using the test data. 
# 

# In[27]:


from sklearn.linear_model import Ridge


# In[28]:


RigeModel = Ridge(alpha=0.1) 
RigeModel.fit(x_train, y_train)
RigeModel.score(x_test, y_test)


# ### Question 10
# Perform a second order polynomial transform on both the training data and testing data. Create and fit a Ridge regression object using the training data, setting the regularisation parameter to 0.1.  Calculate the R^2 utilising the test data provided. Take a screenshot of your code and the R^2.

# In[29]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[features])
x_test_pr=pr.fit_transform(x_test[features])

RigeModel = Ridge(alpha=0.1) 
RigeModel.fit(x_train_pr, y_train)
RigeModel.score(x_test_pr, y_test)


# <p>Once you complete your notebook you will have to share it. Select the icon on the top right a marked in red in the image below, a dialogue box should open, select the option all&nbsp;content excluding sensitive code cells.</p>
#         <p><img width="600" src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/save_notebook.png" alt="share notebook"  style="display: block; margin-left: auto; margin-right: auto;"/></p>
#         <p></p>
#         <p>You can then share the notebook&nbsp; via a&nbsp; URL by scrolling down as shown in the following image:</p>
#         <p style="text-align: center;"><img width="600"  src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/url_notebook.png" alt="HTML" style="display: block; margin-left: auto; margin-right: auto;" /></p>
#         <p>&nbsp;</p>

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

# Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/">Michelle Carey</a>, <a href="www.linkedin.com/in/jiahui-mavis-zhou-a4537814a">Mavis Zhou</a> 

# In[ ]:




