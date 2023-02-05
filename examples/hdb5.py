#!/usr/bin/env python

import numpy as np
from kmodes.kmodes import KModes

# reproduce results on small soybean data set
x = np.genfromtxt('hdb5.csv', dtype=int, delimiter=',')

kmodes_huang = KModes(n_clusters=3, init='Huang', verbose=1)
kmodes_huang.fit(x)

# Print cluster assignment of each input data row (Huang)
clusters_huang = kmodes_huang.fit_predict(x)

print('k-modes (Huang) cluster assignment:')
for r, c in zip(x, clusters_huang):
    print(f"{r} , cluster:{c}")

# Print cluster centroids of the trained model.
print('k-modes (Huang) centroids:')
print(kmodes_huang.cluster_centroids_)

# Print training statistics
print(f'Final training cost: {kmodes_huang.cost_}')
print(f'Training iterations: {kmodes_huang.n_iter_}')

kmodes_cao = KModes(n_clusters=3, init='Cao', verbose=1)
kmodes_cao.fit(x)

# Print cluster assignment of each input data row (Cao)
clusters_cao = kmodes_cao.fit_predict(x)
print('k-modes (Cao) cluster assignment:')
for r2, c2 in zip(x, clusters_cao):
    print(f"{r2} , cluster:{c2}")

# Print cluster centroids of the trained model.
print('k-modes (Cao) centroids:')
print(kmodes_cao.cluster_centroids_)
# Print training statistics
print(f'Final training cost: {kmodes_cao.cost_}')
print(f'Training iterations: {kmodes_cao.n_iter_}')