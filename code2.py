#!/usr/bin/env python
# coding: utf-8

# # 数据集：Wine Reviews
# 2.标称属性：
#     country:Italy 25113, Portugal 6255, US 151136, Spain 11276, France 25184, Argentina 3983
#     province:Sicily & Sardinia 1797, Douro 3072, Oregon 8821,Michigan 219, Northern Spain 3852,Alsace 7457, Rheinhessen 564,Virginia 1951
#   数值属性:
#     points: (min,25%,50%,75%,max):(80,86,,88,,91,100);missing:0
#     price: (min,25%,50%,75%,max):(4,17,25,42,3.3k);missing:8996
#     

# In[1]:


#加载包
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
print(os.listdir("../input"))


# # 数据可视化（直方图，饼图，盒图）
# 直方图显示的是不同组合(points,price)的频数
# 饼图显示的是不同组合(points,price)的频数
# 盒图显示的是不同组合(points,price)的频数

# In[33]:


df= pd.read_csv('../input/winemag-data-130k-v2.csv').fillna(0).head(180000)
#df= pd.DataFrame(pd.Series(list(zip(df['listing_id'],df['price']))),columns=['PriceWithId']).sort_values(by=['PriceWithId'],ascending=True)
df= pd.DataFrame(pd.Series(list(zip(df['points'],df['price']))),columns=['PriceWithpoints'])
df.reset_index(drop=True,inplace=True)
#display(df)

dfNew= df.groupby('PriceWithpoints')
#display(dfNew)
locAccidentSize= np.array([])
for key,value in dfNew:
    locAccidentSize= np.append(locAccidentSize,value.size)
df.drop_duplicates(keep='first',inplace=True)
df.reset_index(drop=True,inplace=True)
locAccidentSize = pd.DataFrame(pd.Series(locAccidentSize),columns=['locAccidentSize'])
dfNewest= pd.concat([df['PriceWithpoints'],locAccidentSize['locAccidentSize']],axis=1).sort_values(by=['locAccidentSize'],ascending= False)
#display(dfNewest[0:10])

plt.figure(figsize=(16,10))
sns.barplot(y=dfNewest['PriceWithpoints'][0:10],x=dfNewest['locAccidentSize'][0:10] , palette='spring')
plt.show()
plt.pie(dfNewest['locAccidentSize'][0:10],labels=dfNewest['PriceWithpoints'][0:10],explode=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1),autopct = '%.2f',shadow=True)
plt.show()
df= pd.read_csv('../input/winemag-data-130k-v2.csv').fillna(0).head(180000)
sns.boxplot(y='price', data=df)
plt.show()


# # 直方图显示price的频率

# In[11]:


df= pd.read_csv('../input/winemag-data-130k-v2.csv').fillna(0).head(180000)
df['price']=df['price'].apply(lambda x : str(x))
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


# # method1:将缺失部分剔除，并且用直方图显示

# In[34]:


lenth=len(df)
df1 = df
for i in range(0,lenth):
    if df1['price'][i] == '0.0':
        df1.drop(index = [i],inplace = True)
        #print(i)
    else:
        pass
df1['price']


# In[25]:


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


# # method2：用最高频率值来填补缺失值,并用直方图显示

# In[35]:


#method2：用最高频率值来填补缺失值
lenth=len(df)
df2 = df
for i in range(0,lenth):
    if df2['price'][i] == '0.0':
        df2['price'][i] = 20.0
        print(i)
    else:
        pass


# In[31]:


df2['price']=df2['price'].apply(lambda x : str(x))
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


# # method3：通过属性的相关关系来填补缺失值（线性回归），并用直方图显示

# In[34]:


#df.info()
df_pred = df[df['price']==0]
#rint(df_pred.shape)
#rint(df_pred)
X_pred = pd.DataFrame(df)
y_pred = df['price']

dataset_train = df1
print(dataset_train.shape)
print(dataset_train)


# In[40]:


dataset_te = dataset_train.drop(['Unnamed: 0'],axis=1)

#使用皮尔逊相关系数：
import numpy
#dataset_train['price']=dataset_train['price'].apply(lambda x : float(x.replace('$','').replace(',','')))
print('相关性:\n', dataset_te.corr())


# In[41]:


from sklearn.linear_model import LinearRegression
line_reg = LinearRegression()
line_reg.fit(dataset_te[['points']],dataset_te[['price']])

print(line_reg.intercept_)
print(line_reg.coef_)


# In[62]:


#开始预测
# X_pred=X_pred.drop(['date'],axis=1)
# X_pred=X_pred.drop(['available'],axis=1)
# X_pred['price']
dataset_pred = pd.DataFrame(['price'])
y_pred = line_reg.predict(X_pred[['points']])
dataset_pred['price'] = pd.DataFrame(y_pred)
dataset_new = dataset_te.append(dataset_pred).sort_values(by='points',axis=0,ascending=True)
dataset_new


# In[63]:


df3 = dataset_te
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


# # mothd4:通过数据对象之间的相似性来填补缺失值（knn），并用直方图显示

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors = 1000, weights = "uniform")

train_knn = df.copy(deep=True).loc[~df['price'].isna(),['points','price']].reset_index(drop=True)
x_knn = train_knn[['points']]
y_knn = train_knn[['price']]
y_knn = y_knn.astype('float').astype('int')
x_knn = x_knn.astype('int')
model_knn.fit(x_knn, np.ravel(y_knn))

# 模型填充
new_dataframe = df.copy(deep=True)
new_dataframe.loc[new_dataframe['price']=='0.0',['price']] = model_knn.predict(new_dataframe.loc[new_dataframe['price']=='0.0',['price']])
#new_dataframe.loc[new_dataframe['price']=='0.0',['price']]
new_dataframe


# In[21]:


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

