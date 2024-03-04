'''
This program implements the gradient-norm-adaptative Hedge algorithm used for
doing the gradient ascent on w.
'''
import numpy as np
import tqdm

from pareto_2d import PC2d
from pareto_nd import PCnd
import matplotlib.pyplot as plt
from random_cloud import *


def hedge_step(w, grad, i, reg):
	η = np.sqrt(np.log(np.size(w, axis=0))) / np.sqrt(1+i) / reg
	w *= np.exp(η*grad)
	w /= np.sum(w)


def hedge(μ, w, M, speedup=False):
	ds, ws, gs = np.zeros((M,) + np.shape(w)), \
		np.zeros((M,) + np.shape(w)), np.zeros(M)
	cloud = PC2d(μ) if speedup else PCnd(μ)
	reg = 0.
	for i in tqdm.tqdm(range(M)):
		g_round, grad_round = cloud.get_cost(w)
		gs[i], ds[i] = g_round, grad_round
		reg = np.maximum(reg, np.max(np.abs(grad_round)))
		hedge_step(w, grad_round, i, reg)
		ws[i] = w
	return gs[-1], np.average(ws, axis=0), ds


if __name__ == '__main__':
	M, K, d = 1000, 10, 3
	μ = draw_points((K,d), width=10.)
	w = np.ones(K)/K
	hedge(μ, w, M)
