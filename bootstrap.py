# -*- coding: utf-8 -*-
"""
Created on Wed May 27 13:31:15 2020

@author: soumya
"""


###################################
# Another easy bootstrap example
###################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

list_normal_numbers = np.random.randn(1000) # standard normal N(0,1)

# now plot distribution of these bootstrapped estimates
sns.distplot(list_normal_numbers)    
np.percentile(list_normal_numbers, 2.5)
np.percentile(list_normal_numbers, 97.5)

lb = np.percentile(list_normal_numbers, 2.5)
ub = np.percentile(list_normal_numbers, 97.5)
sns.distplot(list_normal_numbers)    
plt.vlines(lb, 0, 1)
plt.vlines(ub, 0, 1)
