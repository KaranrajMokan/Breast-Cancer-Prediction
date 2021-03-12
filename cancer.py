#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:22:40 2021

@author: karanrajmokan
"""

import pandas as pd
from texttable import Texttable
import matplotlib.pyplot as plotting
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

breast_cancer = pd.read_csv("data.csv")
breast_cancer = breast_cancer.drop(['Unnamed: 32'],axis=1)
print(breast_cancer.head())
print("\n")



#CHECKING FOR NULL VALUES
full_headers = breast_cancer.columns
values = list(breast_cancer.isnull().sum())
nullList=[]
nullList.append(['Feature','Null Values count'])
for i in range(len(full_headers)):
    nullList.append([full_headers[i],values[i]])

table = Texttable()
table.add_rows(nullList)
print(table.draw())    
print("\n")




#CHECKING FOR CLASS IMBALANCE
imbalance = dict(breast_cancer['diagnosis'].value_counts())
print(imbalance)
breast_cancer.diagnosis.hist(bins=3,grid=False)
plotting.show()
print("\n")




#CHECKING FOR NORMALISATION
print("The histograms of the attributes are given below:")
breast_cancer.hist(bins=5,grid=False,layout=[6,6],figsize=[20,20])
plotting.show()
print("\n")





#CHECKING FOR CORRELATION
print("The correlation heatmap is shown below:")
b = breast_cancer
b_corr = b.drop(['id','diagnosis','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
                 'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'],axis=1)
correlation = b_corr.corr()
heatmap = sns.heatmap(correlation, cbar=True, annot=True, cmap="bwr", linewidths=.75)
heatmap.set_title("Correlation heatmap\n")
plotting.show()
print("\n")




#SPLITTING DATASET FOR TRANINING AND SPLITTING
bc_train, bc_test = train_test_split(breast_cancer, test_size=0.3, random_state=42)
print("After splitting the dataset for training and testing, the shape looks like below")
print("Original Breast Cancer",breast_cancer.shape)
print("Training Breast Cancer",bc_train.shape)
print("Testing Breast Cancer",bc_test.shape)
print("\n")




#BEFORE FEATURE SCALING
breast_cancer['diagnosis'] = breast_cancer.diagnosis.map({'B':0,'M':1})
breast_cancer = breast_cancer.drop(['id'],axis=1)

headers = breast_cancer.columns
minimum = list(map(lambda x: round(x,4),breast_cancer.min()))
mean = list(map(lambda x: round(x,4),breast_cancer.mean()))
maximum = list(map(lambda x: round(x,4),breast_cancer.max()))
std =list(map(lambda x: round(x,4),breast_cancer.std()))

before_scaling=[]
before_scaling.append(['Feature','Min','Mean','Max','Std. Dev'])
for i in range(len(headers)):
    before_scaling.append([headers[i],minimum[i],mean[i],maximum[i],std[i]])

table1 = Texttable()
table1.add_rows(before_scaling)
print(table1.draw())
print("\n")



#AFTER FEATURE SCALING
breast_cancer = pd.DataFrame(preprocessing.scale(breast_cancer.iloc[:,0:32]))

minimum = list(map(lambda x: round(x,4),breast_cancer.min()))
mean = list(map(lambda x: round(x,4),breast_cancer.mean()))
maximum = list(map(lambda x: round(x,4),breast_cancer.max()))
std =list(map(lambda x: round(x,4),breast_cancer.std()))

after_scaling=[]
after_scaling.append(['Feature','Min','Mean','Max','Std. Dev'])
for i in range(len(headers)):
    after_scaling.append([headers[i],minimum[i],mean[i],maximum[i],std[i]])

table2 = Texttable()
table2.add_rows(after_scaling)
print(table2.draw())
print("\n")




#PRINCIPAL COMPONENT ANALYSIS
pca = PCA(n_components=len(breast_cancer.columns))
pca.fit_transform(breast_cancer)
eigen_values = pca.explained_variance_
ratio_values = pca.explained_variance_ratio_
plotting.ylabel("Eigen values")
plotting.xlabel("Number of features")
plotting.title("PCA eigen values")
plotting.ylim(0, max(eigen_values))
plotting.xticks([0,1,2,3,4,5,6,7,8,9,10,15,20,25,30])
plotting.style.context('seaborn-whitegrid')
plotting.axhline(y=1,color='r',linestyle='--')
plotting.plot(eigen_values)
plotting.show()
print("\n")

tableList=[]
tableList.append(["NC","SP","EV","CEV"])
for i in range(len(eigen_values)):
    total=0
    for j in range(i+1):
        total+=ratio_values[j]
    tableList.append([i+1,round(eigen_values[i],2),round(ratio_values[i],2),round(total*100,2)])

table3 = Texttable()
table3.add_rows(tableList)
print(table3.draw())
print("\n")





