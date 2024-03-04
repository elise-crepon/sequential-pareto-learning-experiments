'''
This script was made to test the generic algorithm on random instances.
'''
import time

import numpy as np
import tqdm

from pareto_2d import PC2d
from pareto_nd import PCnd
import matplotlib.pyplot as plt
from random_cloud import draw_spherical


if __name__ == '__main__':
	M = 100
	ds = [2, 3, 4, 5]
	Ks = [4, 8, 16, 32, 64]
	K_max = Ks[-1]
	for d in ds:
		μs = np.concatenate((np.zeros((M,1,d)), draw_spherical((M,K_max,d), width=10.)), axis=1)
		for K in Ks:
			w = np.random.uniform(size=(M, K+1,))
			w /= np.sum(w, axis=1)[:,np.newaxis]
			times = np.zeros(M+1)
			times[0] = time.time()
			for j in tqdm.tqdm(range(M), leave=False, desc=f'd={d} p={K}'):
				_ = PCnd(μs[j,:K+1]).get_cost(w[j])[0]
				times[j+1] = time.time()
			times = np.diff(times)
			tn, σn = np.average(times), np.sqrt(np.var(times))
			print(f'd={d} p={K}: tn:{tn:.2e} (±{σn:.2e})')
