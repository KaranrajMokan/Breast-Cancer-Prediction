#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:22:40 2021

@author: karanrajmokan
"""

import pandas as pd
import matplotlib.pyplot as plotting
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

breast_cancer = pd.read_csv("data.csv")
breast_cancer = breast_cancer.drop(['Unnamed: 32'],axis=1)
print(breast_cancer.head())
print()

#CHECKING FOR NULL VALUES
print(breast_cancer.isnull().sum())
print()

#CHECKING FOR CLASS IMBALANCE
imbalance = dict(breast_cancer['diagnosis'].value_counts())
print(imbalance)
breast_cancer.diagnosis.hist(bins=3,grid=False)
plotting.show()
print()

#CHECKING FOR NORMALISATION
print("The histograms of the attributes are given below:")
breast_cancer.hist(bins=5,grid=False,layout=[6,6],figsize=[20,20])
plotting.show()
print()


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
print()

