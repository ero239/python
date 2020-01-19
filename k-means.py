from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

"""
Data Management
"""
data = pd.read_csv("gapminder_ghana_updated.csv")


# Data Management

data_clean = data.dropna()

# subset clustering variables
cluster=data_clean[['incomeperperson','exports','malebloodpressure',
'totalpopulationfemale','inflation','agriculture','democracyscore',
'agriculturalland','aidreceived','aidreceivedperperson','realgdppercapita',
'femalebloodpressure','malebodymassindex','femalebodymassindex','underfivemortality',
'totalfertilityrate','cholesterolinmen','cholesterolinwomen','crudebirthrate', 
'deadkidsperwoman','externaldebtstocks','energyuse']]



cluster.describe()

# standardize clustering variables to have mean=0 and sd=1
clustervar=cluster.copy()
clustervar['incomeperperson']=preprocessing.scale(clustervar['incomeperperson'].astype('float64'))
clustervar['exports']=preprocessing.scale(clustervar['exports'].astype('float64'))
clustervar['malebloodpressure']=preprocessing.scale(clustervar['malebloodpressure'].astype('float64'))
clustervar['totalpopulationfemale']=preprocessing.scale(clustervar['totalpopulationfemale'].astype('float64'))
clustervar['inflation']=preprocessing.scale(clustervar['inflation'].astype('float64'))
clustervar['agriculture']=preprocessing.scale(clustervar['agriculture'].astype('float64'))
clustervar['democracyscore']=preprocessing.scale(clustervar['democracyscore'].astype('float64'))
clustervar['agriculturalland']=preprocessing.scale(clustervar['agriculturalland'].astype('float64'))
clustervar['aidreceived']=preprocessing.scale(clustervar['aidreceived'].astype('float64'))
clustervar['aidreceivedperperson']=preprocessing.scale(clustervar['aidreceivedperperson'].astype('float64'))
clustervar['realgdppercapita']=preprocessing.scale(clustervar['realgdppercapita'].astype('float64'))
clustervar['femalebloodpressure']=preprocessing.scale(clustervar['femalebloodpressure'].astype('float64'))
clustervar['malebodymassindex']=preprocessing.scale(clustervar['malebodymassindex'].astype('float64'))
clustervar['femalebodymassindex']=preprocessing.scale(clustervar['femalebodymassindex'].astype('float64'))
clustervar['underfivemortality']=preprocessing.scale(clustervar['underfivemortality'].astype('float64'))
clustervar['totalfertilityrate']=preprocessing.scale(clustervar['totalfertilityrate'].astype('float64'))
clustervar['cholesterolinmen']=preprocessing.scale(clustervar['cholesterolinmen'].astype('float64'))
clustervar['cholesterolinwomen']=preprocessing.scale(clustervar['cholesterolinwomen'].astype('float64'))
clustervar['crudebirthrate']=preprocessing.scale(clustervar['crudebirthrate'].astype('float64'))
clustervar['deadkidsperwoman']=preprocessing.scale(clustervar['deadkidsperwoman'].astype('float64'))
clustervar['externaldebtstocks']=preprocessing.scale(clustervar['externaldebtstocks'].astype('float64'))
clustervar['energyuse']=preprocessing.scale(clustervar['energyuse'].astype('float64'))


# split data into train and test sets
clus_train, clus_test = train_test_split(clustervar, test_size=.3, random_state=123)

# k-means cluster analysis for 1-9 clusters                                                           
from scipy.spatial.distance import cdist
clusters=range(1,10)
meandist=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clus_train)
    clusassign=model.predict(clus_train)
    meandist.append(sum(np.min(cdist(clus_train, model.cluster_centers_, 'euclidean'), axis=1)) 
    / clus_train.shape[0])

"""
Plot average distance from observations from the cluster centroid
to use the Elbow Method to identify number of clusters to choose
"""

plt.plot(clusters, meandist)
plt.xlabel('Number of clusters')
plt.ylabel('Average distance')
plt.title('Selecting k with the Elbow Method')


# Interpret 3 cluster solution
model3=KMeans(n_clusters=3)
model3.fit(clus_train)
clusassign=model3.predict(clus_train)
# plot clusters

from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(clus_train)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=model3.labels_,)
plt.xlabel('Canonical variable 1')
plt.ylabel('Canonical variable 2')
plt.title('Scatterplot of Canonical Variables for 3 Clusters')
plt.show()

"""
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""
# create a unique identifier variable from the index for the 
# cluster training data to merge with the cluster assignment variable
clus_train.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslist=list(clus_train['index'])
# create a list of cluster assignments
labels=list(model3.labels_)
# combine index variable list with cluster assignment list into a dictionary
newlist=dict(zip(cluslist, labels))
newlist
# convert newlist dictionary to a dataframe
newclus=DataFrame.from_dict(newlist, orient='index')
newclus
# rename the cluster assignment column
newclus.columns = ['cluster']

# now do the same for the cluster assignment variable
# create a unique identifier variable from the index for the 
# cluster assignment dataframe 
# to merge with cluster training data
newclus.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_train=pd.merge(clus_train, newclus, on='index')
merged_train.head(n=100)
# cluster frequencies
merged_train.cluster.value_counts()

"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# FINALLY calculate clustering variable means by cluster
clustergrp = merged_train.groupby('cluster').mean()
print ("Clustering variable means by cluster")
print(clustergrp)


# validate clusters in training data by examining cluster differences in lifeexpectancy using ANOVA
# first have to merge lifeexpectancy with clustering variables and cluster assignment data 
gpa_data=data_clean['lifeexpectancy']
# split lifeexpectancy data into train and test sets
gpa_train, gpa_test = train_test_split(gpa_data, test_size=.3, random_state=123)
gpa_train1=pd.DataFrame(gpa_train)
gpa_train1.reset_index(level=0, inplace=True)
merged_train_all=pd.merge(gpa_train1, merged_train, on='index')
sub1 = merged_train_all[['lifeexpectancy', 'cluster']].dropna()

import statsmodels.formula.api as smf
import statsmodels.stats.multicomp as multi 

lfemod = smf.ols(formula='lifeexpectancy ~ C(cluster)', data=sub1).fit()
print (lfemod.summary())

print ('means for lifeexpectancy by cluster')
m1= sub1.groupby('cluster').mean()
print (m1)

print ('standard deviations for lifeexpectancy by cluster')
m2= sub1.groupby('cluster').std()
print (m2)

mc1 = multi.MultiComparison(sub1['lifeexpectancy'], sub1['cluster'])
res1 = mc1.tukeyhsd()
print(res1.summary())









"""
VALIDATION
BEGIN multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster in test data set
"""
# create a variable out of the index for the cluster training dataframe to merge on
clus_test.reset_index(level=0, inplace=True)
# create a list that has the new index variable
cluslistval=list(clus_test['index'])
# create a list of cluster assignments
#labelsval=list(clusassignval)
labelsval=list(cluslistval)
# combine index variable list with labels list into a dictionary
#newlistval=dict(zip(cluslistval, clusassignval))
newlistval=dict(zip(cluslistval, labelsval))
newlistval
# convert newlist dictionary to a dataframe
newclusval=DataFrame.from_dict(newlistval, orient='index')
newclusval
# rename the cluster assignment column
newclusval.columns = ['cluster']
# create a variable out of the index for the cluster assignment dataframe to merge on
newclusval.reset_index(level=0, inplace=True)
# merge the cluster assignment dataframe with the cluster training variable dataframe
# by the index variable
merged_test=pd.merge(clus_test, newclusval, on='index')
# cluster frequencies
merged_test.cluster.value_counts()
"""
END multiple steps to merge cluster assignment with clustering variables to examine
cluster variable means by cluster
"""

# calculate test data clustering variable means by cluster
clustergrpval = merged_test.groupby('cluster').mean()
print ("Test data clustering variable means by cluster")
print(clustergrpval)
