#!/usr/bin/env ipython

import numpy as np
import numpy.random as npr

sample = np.hstack([npr.randn(20)*4+10, npr.randn(15)*2-5])
rightsplit = np.hstack([np.zeros(20), np.ones(15)])

o = sample.argsort()

sample = sample[o]
rightsplit = rightsplit[o]

import cluster

cl = cluster.HierarchicalClustering(sample, lambda x,y: abs(x-y))
print(cl.getlevel(1))     # get clusters of items closer than 10



