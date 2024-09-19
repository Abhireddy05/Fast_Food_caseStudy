#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[5]:


df=pd.read_csv("mcdonalds.csv")
print(df.columns.tolist())


# In[6]:


df.shape


# In[7]:


df.head(n=3)


# In[8]:


df.describe()


# In[9]:


import numpy as np
df_cols=df.iloc[:,0:11]
df_cols = (df_cols == "Yes").astype(int)
column_means = df_cols.mean().round(2)
print(column_means)


# In[10]:


df_cols.head()


# In[11]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df_cols.hist(figsize=(5,5))


# In[12]:


df_cols.corr()


# In[13]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


# In[14]:


df.isnull().sum()


# In[15]:



from sklearn.preprocessing import LabelEncoder


label_encoder = LabelEncoder()


for column in df.select_dtypes(include=['object']):
    df[column] = label_encoder.fit_transform(df[column])


# In[16]:


df.dtypes


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df_cols.hist(figsize=(20,30))


# In[24]:


plt.figure(figsize=(12,9))
sns.heatmap(df_cols.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# In[25]:



plt.figure(figsize=(20,10))
sns.countplot(data=df, x='Age', hue='Gender', palette='Set2')
plt.xlabel("Age")
plt.ylabel("Person Count")
plt.title("Age Distribution by Gender")
plt.show()


# In[26]:


attributes = ['yummy', 'convenient', 'spicy', 'fattening', 'greasy', 'fast', 'cheap', 'tasty', 'expensive', 'healthy', 'disgusting']
mean_values = df.groupby('Gender')[attributes].mean().reset_index()

plt.figure(figsize=(20,10))
mean_values.set_index('Gender').plot(kind='bar', stacked=False, colormap='Paired')
plt.xlabel("Gender")
plt.ylabel("Average Rating")
plt.title("Average Ratings of Attributes by Gender")
plt.legend(loc='upper right', title='Attributes')
plt.show()


# In[27]:



df.head()


# In[28]:



features = df_cols


# In[29]:


scaler = StandardScaler()


# In[30]:


scaled_features = scaler.fit_transform(features)
pca = PCA()
pca_result = pca.fit_transform(scaled_features)
explained_variance = pca.explained_variance_ratio_
explained_variance


# In[31]:


cumulative_variance = np.cumsum(explained_variance)
cumulative_variance


# In[32]:


pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
pca_df


# In[33]:


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()


# In[37]:


sns.pairplot(pca_df[['PC1', 'PC2', 'PC3', 'PC4','PC5', 'PC6', 'PC7', 'PC8','PC9', 'PC10', 'PC11']], diag_kind='kde')
plt.suptitle('Pairwise PCA Scatter Plots', y=1.02)
plt.show()


# In[35]:


sns.pairplot(df_cols)


# In[38]:


from sklearn.cluster import KMeans
     

kmeans = KMeans(n_clusters=12)
pca_df['Cluster'] = kmeans.fit_predict(pca_result)


# In[39]:


plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='tab10')
plt.title('Clusters in PCA Space')
plt.show()


# In[40]:


from scipy.spatial.distance import pdist, squareform

distance_matrix = pdist(pca_df, metric='euclidean')
distance_matrix = squareform(distance_matrix)



     

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

linkage_matrix = linkage(distance_matrix, method='ward')


# In[41]:


plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=pca_df.index, leaf_rotation=90, leaf_font_size=10)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()


# In[42]:


num_clusters = 11


pca_df['Cluster'] = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

     

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='tab10')
plt.title('Hierarchical Clustering in PCA Space')
plt.show()


# In[43]:


sns.boxplot(data=df_cols)


# In[ ]:




