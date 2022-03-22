#!/usr/bin/env python
# coding: utf-8

# # 数据集：Melbourne Airbnb Open Data
# 1.数值属性:
#     price:(min,25%,50%,75%,max):();mising:5.26m

# In[1]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
print(os.listdir("../input"))


# # 数据可视化（直方图，饼图，盒图）
# 直方图显示的是不同组合(ID,price)的频数 
# 饼图显示的是不同组合(ID,price)的频数 
# 盒图显示的是price

# In[4]:


df= pd.read_csv('../input/calendar_dec18.csv').fillna(0).head(180000)
#df= pd.DataFrame(pd.Series(list(zip(df['listing_id'],df['price']))),columns=['PriceWithId']).sort_values(by=['PriceWithId'],ascending=True)
df= pd.DataFrame(pd.Series(list(zip(df['listing_id'],df['price']))),columns=['PriceWithId'])
df.reset_index(drop=True,inplace=True)
#display(df)

dfNew= df.groupby('PriceWithId')
#display(dfNew)
locAccidentSize= np.array([])
for key,value in dfNew:
    locAccidentSize= np.append(locAccidentSize,value.size)
df.drop_duplicates(keep='first',inplace=True)
df.reset_index(drop=True,inplace=True)
locAccidentSize = pd.DataFrame(pd.Series(locAccidentSize),columns=['locAccidentSize'])
dfNewest= pd.concat([df['PriceWithId'],locAccidentSize['locAccidentSize']],axis=1).sort_values(by=['locAccidentSize'],ascending= False)
#display(dfNewest[0:10])

plt.figure(figsize=(16,10))
sns.barplot(y=dfNewest['PriceWithId'][0:1000],x=dfNewest['locAccidentSize'][0:1000] , palette='spring')
plt.show()
plt.pie(dfNewest['locAccidentSize'][0:10],labels=dfNewest['PriceWithId'][0:10],explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1),autopct = '%.2f',shadow=True)
plt.show()


# In[9]:


df= pd.read_csv('../input/calendar_dec18.csv').fillna(0).head(180000) 
lenth=len(df)
for i in range(0,lenth):
    if df['price'][i] == 0:
        df['price'][i] = str(df['price'][i])
        print(i)

df['price']=df['price'].apply(lambda x : float(x.replace('$','').replace(',','')))
sns.boxplot(y='price', data=df) 
plt.show()


# 做柱状图比较不同price数值的频率，由高到低排列。

# # 直方图显示price的频数

# In[164]:


df= pd.read_csv('../input/calendar_dec18.csv').fillna(0).head(180000)
dfAccType = pd.DataFrame(df['price'])
#display(dfAccType)
dfGrouped = dfAccType.groupby('price')
listArray  = np.array([])
listNames = pd.DataFrame(dfAccType['price'].unique(),index=list(dfAccType['price'].unique()),columns=['price'])
#display(listNames)
listSize= pd.DataFrame(pd.value_counts(dfAccType['price'])).rename(columns={'price':'Count'})
#display(listSize)
dfNewest = pd.concat([listNames,listSize],axis=1,sort=True).sort_values(by=('Count'),ascending=False).reset_index(drop=True,inplace=False)
display(dfNewest[0:10])
plt.figure(figsize=(16,6))
sns.barplot(y=dfNewest['price'][0:10],x=dfNewest['Count'][0:10],palette='autumn')
plt.show()


# # method1：将缺失部分剔除

# In[10]:


lenth=len(df)
df1 = df
for i in range(0,lenth):
    
    if df1['price'][i] == 0:
        df1.drop(index = [i],inplace = True)
        print(i)
    else:
        pass


# 做将缺失部分剔除后的柱状图

# In[99]:


df1AccType = pd.DataFrame(df1['price'])
#display(dfAccType)
df1Grouped = df1AccType.groupby('price')
listArray  = np.array([])
listNames = pd.DataFrame(df1AccType['price'].unique(),index=list(df1AccType['price'].unique()),columns=['price'])
#display(listNames)
listSize= pd.DataFrame(pd.value_counts(df1AccType['price'])).rename(columns={'price':'Count'})
#display(listSize)
df1Newest = pd.concat([listNames,listSize],axis=1,sort=True).sort_values(by=('Count'),ascending=False).reset_index(drop=True,inplace=False)
display(df1Newest[0:10])
plt.figure(figsize=(16,6))
sns.barplot(y=df1Newest['price'][0:10],x=df1Newest['Count'][0:10],palette='autumn')
plt.show()


# 可以看出柱状图少了price=0的情况。

# # method2：用最高频率值来填补缺失值

# In[11]:


lenth=len(df)
df2 = df
for i in range(0,lenth):
    if df2['price'][i] == 0:
        df2['price'][i] = 150.00
        print(i)
    else:
        pass


# In[106]:


df2AccType = pd.DataFrame(df2['price'])
#display(dfAccType)
df2Grouped = df2AccType.groupby('price')
listArray  = np.array([])
listNames = pd.DataFrame(df2AccType['price'].unique(),index=list(df2AccType['price'].unique()),columns=['price'])
#display(listNames)
listSize= pd.DataFrame(pd.value_counts(df2AccType['price'])).rename(columns={'price':'Count'})
#display(listSize)
df2Newest = pd.concat([listNames,listSize],axis=1,sort=True).sort_values(by=('Count'),ascending=False).reset_index(drop=True,inplace=False)
display(df2Newest[0:10])
plt.figure(figsize=(16,6))
sns.barplot(y=df2Newest['price'][0:10],x=df2Newest['Count'][0:10],palette='autumn')
plt.show()


# 用频数最高的150代替缺失值后，150的频数依然是最高的。

# # method3：通过属性的相关关系来填补缺失值

# In[178]:


#df.info()
df_pred = df[df['price']==0]
#rint(df_pred.shape)
#rint(df_pred)
X_pred = pd.DataFrame(df)
y_pred = df['price']

dataset_train = df1
print(dataset_train.shape)
print(dataset_train)


# In[230]:


#dataset_te = dataset_train.drop(['listing_id'],axis=1)
#使用皮尔逊相关系数：
import numpy
dataset_train['price']=dataset_train['price'].apply(lambda x : float(x.replace('$','').replace(',','')))
print('相关性:\n', dataset_train.corr())


# In[242]:


from sklearn.linear_model import LinearRegression
line_reg = LinearRegression()
line_reg.fit(dataset_train[['listing_id']],dataset_train[['price']])

print(line_reg.intercept_)
print(line_reg.coef_)


# In[305]:


#开始预测
# X_pred=X_pred.drop(['date'],axis=1)
# X_pred=X_pred.drop(['available'],axis=1)
# X_pred.head
y_pred = line_reg.predict(X_pred[['listing_id']])
dataset_pred['price'] = pd.DataFrame(y_pred)
dataset_new = dataset_train.append(dataset_pred).sort_values(by='listing_id',axis=0,ascending=True)
dataset_new


# In[314]:


df3 = dataset_new
df3['price']=df3['price'].apply(lambda x : str(x))
#print(type(df3['price'][300]))
df3AccType = pd.DataFrame(df3['price'])
#display(dfAccType)
df3Grouped = df3AccType.groupby('price')
listArray  = np.array([])
listNames = pd.DataFrame(df3AccType['price'].unique(),index=list(df3AccType['price'].unique()),columns=['price'])
#display(listNames)
listSize= pd.DataFrame(pd.value_counts(df3AccType['price'])).rename(columns={'price':'Count'})
#display(listSize)
df3Newest = pd.concat([listNames,listSize],axis=1,sort=True).sort_values(by=('Count'),ascending=False).reset_index(drop=True,inplace=False)
display(df3Newest[0:10])
plt.figure(figsize=(16,6))
sns.barplot(y=df3Newest['price'][0:10],x=df3Newest['Count'][0:10],palette='autumn')
plt.show()


# # methd4:通过数据对象之间的相似性来填补缺失值

# In[359]:


from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors = 1000, weights = "uniform")

train_knn = df.copy(deep=True).loc[~df['price'].isna(),['listing_id','price']].reset_index(drop=True)
x_knn = train_knn[['listing_id']]
y_knn = train_knn[['price']]
y_knn=y_knn.astype('int')
model_knn.fit(x_knn, np.ravel(y_knn))

# 模型填充
new_dataframe = df.copy(deep=True)
new_dataframe.loc[new_dataframe['price'].isna(),['price']] = model_knn.predict(new_dataframe.loc[new_dataframe['price'].isna(),['listing_id']])

new_dataframe


# In[360]:


df4 = new_dataframe
df4['price']=df4['price'].apply(lambda x : str(x))
#print(type(df3['price'][300]))
df4AccType = pd.DataFrame(df4['price'])
#display(dfAccType)
df4Grouped = df4AccType.groupby('price')
listArray  = np.array([])
listNames = pd.DataFrame(df4AccType['price'].unique(),index=list(df4AccType['price'].unique()),columns=['price'])
#display(listNames)
listSize= pd.DataFrame(pd.value_counts(df4AccType['price'])).rename(columns={'price':'Count'})
#display(listSize)
df4Newest = pd.concat([listNames,listSize],axis=1,sort=True).sort_values(by=('Count'),ascending=False).reset_index(drop=True,inplace=False)
display(df4Newest[0:10])
plt.figure(figsize=(16,6))
sns.barplot(y=df4Newest['price'][0:10],x=df4Newest['Count'][0:10],palette='autumn')
plt.show()

