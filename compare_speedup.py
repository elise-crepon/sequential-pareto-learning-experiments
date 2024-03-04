'''
This script was made to compare the computing time used by the generic algorithm
and the one specifically designed for dimension 2. It was used to generate
Table 1.
'''
import time

import numpy as np
import tqdm

from pareto_2d import PC2d
from pareto_nd import PCnd
import matplotlib.pyplot as plt
from random_cloud import draw_only_pareto


if __name__ == '__main__':
	M, d, lδ = 1000, 2, -np.log(1e-2)
	μ_front = np.zeros((M,0,d))
	for K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
		while np.size(μ_front, axis=1) < K:
			μ_front = draw_only_pareto(μ_front, width=10.)
		μ = np.concatenate((np.zeros((M,1,d)), μ_front), axis=1)
		w = np.random.uniform(size=(M, K+1,))
		w /= np.sum(w, axis=1)[:,np.newaxis]
		costs = np.zeros((M,2))
		times = np.zeros((2,M+1))
		start = time.time()
		times[0,0] = time.time()
		for j in tqdm.tqdm(range(M), leave=False, desc='2d'):
			pc2d = PC2d(μ[j])
			costs[j,0] = pc2d.get_cost(w[j])[0]
			times[0,j] = time.time()
		times[0,M] = time.time()
		for j in tqdm.tqdm(range(M), leave=False, desc='nd'):
			pcnd = PCnd(μ[j])
			costs[j,1] = pcnd.get_cost(w[j])[0]
			times[1,j] = time.time()
		times[1,M] = time.time()
		end = time.time()
		times = np.diff(times, axis=1)
		tn, t2 = np.average(times[1]), np.average(times[0])
		σn, σ2 = np.sqrt(np.var(times[1])), np.sqrt(np.var(times[0]))
		Δ = np.abs(costs[:,1] - costs[:,0])
		assert(np.all(Δ < 1e-12))
		print(f'{K}: {tn/t2:.3} (tn:{tn:.2e} (±{σn:.2e}), t2:{t2:.2e} (±{σ2:.2e})) → {end-start:.1f}')
