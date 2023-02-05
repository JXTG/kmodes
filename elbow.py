#!/usr/bin/env python

# importing necessary libraries
import pandas as pd
import numpy as np
# !pip install kmodes
from kmodes.kmodes import KModes
import matplotlib.pyplot as plt
# matplotlib inline

x = pd.read_csv('hdb5.csv')

# Elbow curve to find optimal K
cost = []
K = range(1,10)
for num_clusters in list(K):
#    kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
#    kmode = KModes(n_clusters=num_clusters, init='Huang', verbose=1)
    kmode = KModes(n_clusters=num_clusters, init='Cao', verbose=1)
    kmode.fit_predict(x)
    cost.append(kmode.cost_)

plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
#plt.title('Elbow Method For Optimal k, using 5 params and "random"')
#plt.title('Elbow Method For Optimal k, using 5 params and "Huang"')
plt.title('Elbow Method For Optimal k, using 5 params and "Cao"')
plt.show()

# from soybean example:
# KModes(n_clusters=4, init='Huang', verbose=1)