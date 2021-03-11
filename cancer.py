#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 18:22:40 2021

@author: karanrajmokan
"""

import pandas as pd
import matplotlib.pyplot as plotting
import warnings
warnings.filterwarnings("ignore")

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

