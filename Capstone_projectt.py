#!/usr/bin/env python
# coding: utf-8

# #  Q2. Download the CAR DETAILS dataset and perform Data cleaning
# ## and Data Pre-Processing if Necessary

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


df=pd.read_excel('CAR_DETAILS.xlsx')
df


# In[19]:


df.describe()


# In[20]:


df['name'].unique()


# In[21]:


df['Company_name']=df['name'].apply(lambda x:" ".join(x.split()[0:3]))


# In[22]:


def fetch_processor(text):
    if text.split()[0] == 'Ford':
        return 'Ford'
    if text.split()[0] == 'Maruti':
        return 'Maruti'
    if text.split()[0] == 'Honda':
        return 'Honda'
    if text.split()[0] == 'Mahindra':
        return 'Mahindra'
    if text.split()[0] == 'Tata':
        return 'Tata'
    if text.split()[0] == 'Hyundai':
        return 'Hyundai'
    if text.split()[0] == 'Toyota':
        return 'Toyota'
    if text.split()[0] == 'Datsun':
        return 'Datsun'
    if text.split()[0] == 'Chevrolet':
        return 'Chevrolet'
    if text.split()[0] == 'Jaguar':
        return 'Jaguar'


# In[23]:


df['Company_name']=df['name'].apply(fetch_processor)


# In[24]:


df.head(5)


# In[25]:


df['Company_name'].unique()


# In[26]:


column_titles = df.columns.tolist()
column_titles


# # Checking relationship of Company with Price
# 

# In[27]:


plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='Company_name',y='selling_price',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# # Checking relationship of Year with Price
# 

# In[28]:


plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='selling_price',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[29]:


df.drop('Company_name', axis=1, inplace=True)


# # Checking relationship of kms_driven with Price
# 

# In[ ]:


sns.relplot(x='km_driven',y='selling_price',data=df,height=7,aspect=1.5)


# # Checking relationship of Fuel Type with Price
# 

# In[ ]:


plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel',y='selling_price',data=df)


# # Relationship of Price with FuelType, Year and Company mixed
# 

# In[ ]:


ax=sns.relplot(x='Company_name',y='selling_price',data=df,hue='fuel',size='year',height=7,aspect=2)
ax.set_xticklabels(rotation=40,ha='right')


# In[ ]:


df


# In[ ]:


df.info()


# In[ ]:


df.dtypes


# In[ ]:


df.shape


# In[ ]:


df.fuel.unique()


# In[33]:


df.isnull().sum()


# In[34]:


df.name.unique()


# In[35]:


# Assuming df is your DataFrame and 'column_name' is the name of the column you want to count unique values for
unique_values_count = df['name'].value_counts()

print(unique_values_count)


# In[36]:


df.transmission.unique()


# In[37]:


df.seller_type.unique()


# In[38]:


df.owner.unique()


# # Data Preprocessing

# # Handling Null Values

# In[39]:


df.isnull().sum()


# In[40]:


df.duplicated().sum()


# # Separate continous and categorical features

# In[41]:


cat_cols=df.dtypes[df.dtypes=='object'].index
num_cols=df.dtypes[df.dtypes!='object'].index
print(cat_cols)
print(num_cols)


# # Exploring and cleaning categorical features

# In[42]:


for i in cat_cols:
    print(f'Feature:{i}, | Count of Unique: {df[i].nunique()}')
    print(f'value_counts{df[i].unique()}')
    print('*'*50)


# In[ ]:





# In[43]:


df


# # Cleaning Extra Spaces

# In[44]:


for i in cat_cols:
    df[i]=df[i].apply(lambda x:x.strip())


# In[45]:


for i in cat_cols:
    print(f'Feature:{i},  | Count of Unique:{df[i].nunique()}')
    print(f'value_counts{df[i].unique()}')
    print('*'*50)


# In[46]:


print(cat_cols)


# In[47]:


df['name'].value_counts()


# In[48]:


df['fuel'].value_counts()


# In[49]:


df['seller_type'].value_counts()


# In[50]:


df['transmission'].value_counts()


# In[51]:


df['owner'].value_counts()


# #  Q3. Use the various methods such as Handling null values, One-Hot
# ## Encoding, Imputation and Scaling of Data Pre-Processing where
# ## necessary.

# In[52]:


from sklearn.preprocessing import LabelEncoder


# In[53]:


df['fuel'].value_counts()


# In[54]:


le_fuel=LabelEncoder()
df['fuel']=le_fuel.fit_transform(df['fuel'])


# In[55]:


df['seller_type'].unique()


# In[56]:


df['seller_type'].value_counts()


# In[57]:


le_seller_type=LabelEncoder()
df['seller_type']=le_seller_type.fit_transform(df['fuel'])


# In[58]:


df['transmission'].unique()


# In[59]:


df['transmission'].value_counts()


# In[60]:


le_transmission=LabelEncoder()
df['transmission']=le_transmission.fit_transform(df['transmission'])


# In[61]:


df['owner'].unique()


# In[62]:


df['owner'].value_counts()


# In[63]:


le_owner=LabelEncoder()
df['owner']=le_owner.fit_transform(df['owner'])


# In[64]:


title=[]
for i in df['name']:
  title.append(i.split()[0])


# In[65]:


title=np.array(title)
title=pd.Series(title,name='Title')


# In[66]:


df=pd.concat([df,title],axis=1)


# In[67]:


df.drop('name',inplace=True,axis=1)


# In[68]:


df['Title'].unique()


# In[69]:


df['Title'].value_counts()


# In[70]:


le_Title=LabelEncoder()
df['Title']=le_Title.fit_transform(df['Title'])


# In[71]:


df


# ## Q4. Perform Exploratory data analysis (EDA) on the Data and perform
# ## Graphical Analysis on the Data. Include the graphs with
# ## conclusions from the Graphical Analysis.

# In[72]:


cat_cols=df.dtypes[df.dtypes=='object'].index
print(cat_cols)


# In[73]:


corr=df.corr(numeric_only=True)
plt.figure(figsize=(7,7))
sns.heatmap(corr,annot=True, cmap='coolwarm')
plt.show()


# In[74]:


print(num_cols)


# In[75]:


plt.figure(figsize=(20,20))
for i in range(len(num_cols)):
    plt.subplot(2,3,i+1)
    sns.boxplot(x=df[num_cols[i]])
    plt.title(f'Boxplot for {num_cols[i]}')
plt.show()


# In[76]:


df.head()


# In[77]:


a=df[num_cols].describe(percentiles=[0.01,0.02,0.98,0.99]).T
a=a.iloc[:,3:]
a


# In[78]:


sns.pairplot(df,diag_kind="kde", diag_kws=dict(shade=True, bw=.05, vertical=False))
plt.show()


# In[ ]:





# In[79]:


plt.subplots(figsize=(12,5))
ax=sns.barplot(x='year',y='selling_price',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()


# In[80]:


sns.relplot(x='km_driven',y='selling_price',data=df,height=5,aspect=1.5)


# In[81]:


df


# In[82]:


df.drop('Title', axis=1, inplace=True)


# In[83]:


df


# ### Q5. Prepare the Data for Machine Learning modeling.

# In[84]:


#### Select x and y


# In[85]:


x=df.drop('selling_price',axis=1)
y=df['selling_price']
print(x.shape)
print(y.shape)


# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=25)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[88]:


#### Create a function to evaluate the model


# In[89]:


from sklearn.metrics import *


# In[90]:


def eval_model(model,x_train,y_train,x_test,y_test,mname):
    model.fit(x_train,y_train)
    ypred = model.predict(x_test)
    train_r2 = model.score(x_train,y_train)
    test_r2 = model.score(x_test,y_test)
    test_mae = mean_absolute_error(y_test,ypred)
    test_mse = mean_squared_error(y_test,ypred)
    test_rmse = np.sqrt(test_mse)
    res = pd.DataFrame({'Train_R2':train_r2,'Test_R2':test_r2,
                        'Test_MSE':test_mse,'Test_RMSE':test_rmse,'Test_MAE':test_mae},
                       index=[mname])
    return res


# ### Q6. Apply various Machine Learning techniques such as Regression or
# ## classification ,Bagging, Ensemble techniques and find out the
# ## best model using various Machine Learning model evaluation
# ## metrics.

# In[91]:


from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor


# In[92]:


#### Model Building


# In[93]:


#### 1) Lin Reg


# In[94]:


lr1 = LinearRegression()
lr_res = eval_model(lr1,x_train,y_train,x_test,y_test,'Lin_Reg')
lr_res


# In[95]:


#### 2) Ridge Regression


# In[96]:


rid = Ridge(alpha=0.2)
rid_res = eval_model(rid,x_train,y_train,x_test,y_test,'Ridge')
rid_res


# In[97]:


#### 3) Lasso


# In[98]:


las = Lasso(alpha=12)
las_res = eval_model(las,x_train,y_train,x_test,y_test,'Lasso')
las_res


# In[99]:


#### 4) DT Reg


# In[100]:


dt1 = DecisionTreeRegressor(max_depth=5,min_samples_split=10)
dt_res = eval_model(dt1,x_train,y_train,x_test,y_test,'DT_Reg')
dt_res


# In[101]:


#### 5) AdaBoost Reg


# In[102]:


adab = AdaBoostRegressor(n_estimators=80,random_state=42)
adab_res = eval_model(adab,x_train,y_train,x_test,y_test,'Adab_Reg')
adab_res


# In[103]:


#### 6) RF Reg


# In[104]:


rf1 = RandomForestRegressor(n_estimators=100,max_depth=4,min_samples_split=8)
rf_res = eval_model(rf1,x_train,y_train,x_test,y_test,'RF_Reg')
rf_res


# In[105]:


#### 7) KNN Reg


# In[106]:


knn = KNeighborsRegressor(n_neighbors=7)
knn_res = eval_model(knn,x_train,y_train,x_test,y_test,'KNN_Reg')
knn_res


# In[107]:


#### Tabulate All Results


# In[108]:


res = pd.concat([lr_res,rid_res,las_res, dt_res,adab_res,rf_res,knn_res])
res


# ### Q7. Save the best model and Load the model.

# #### Save the model

# In[109]:


res = pd.concat([lr_res,rid_res,las_res, dt_res,adab_res,rf_res,knn_res])
res


# ### On the basis of RMSE, Decision Tree is the best Algorithm

# In[110]:


dt1.score(x_train,y_train)


# In[111]:


import pickle


# In[112]:


pickle.dump(dt1,open('dt1_modell.pkl','wb'))


# In[113]:


pickle.dump(lr1,open('lr1_modell.pkl','wb'))


# In[114]:


pickle.dump(rf1,open('rf1_modell.pkl','wb'))


# ## Q8. Take the original data set and make another dataset by randomly
# ## picking 20 data points from the CAR DETAILS dataset and apply
# ## the saved model on the same Dataset and test the model.
# 

# In[115]:


with open('dt1_modell.pkl','rb') as file:
    pipeline =pickle.load(file)


# In[116]:


with open('lr1_modell.pkl','rb') as file:
    pipeline =pickle.load(file)


# In[117]:


with open('rf1_modell.pkl','rb') as file:
    pipeline =pickle.load(file)


# In[118]:


sample=df.sample(20)
sample.to_csv("Sample.csv")

y_sample_true=sample['selling_price']
sample=sample.drop('selling_price',axis=1)


# In[119]:


dt1.predict(sample)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




