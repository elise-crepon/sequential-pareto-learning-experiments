'''
This programs implements the Track-and-Stop algorithm.
'''
import numpy as np
import matplotlib.pyplot as plt

from pareto_2d import PC2d
from pareto_nd import PCnd
from hedge import hedge, hedge_step
from random_cloud import draw_points


def track_and_stop(μ, lδ, tracking='D', \
		T=None, silent=False, speedup=False):
	creator = lambda λ: PC2d(λ) if speedup else PCnd(λ)
	K = np.size(μ, axis=0)

	Σ_hat, N_hat, t = np.zeros(np.shape(μ)), np.zeros(K), 0
	N_hedge, w_hedge, reg = np.ones(K), np.ones(K)/K, 0.

	# Estimate problem difficulty
	if T is None:
		g, _, _ = hedge(μ, w_hedge.copy(), 300)
		T = int(np.ceil(lδ / g))

	# Create first estimate for Σ_hat
	for k in range(K):
		Σ_hat[k] = np.random.normal(μ[k])
		t += 1
		N_hat[k] += 1
	μ_hat = np.copy(Σ_hat)
	cloud_hat = creator(μ_hat)

	while True:
		g_round, grad_round = cloud_hat.get_cost(w_hedge)
		reg = np.maximum(reg, np.max(np.abs(grad_round)))
		hedge_step(w_hedge, grad_round, t, reg)

		if tracking == 'C': N_hedge += w_hedge
		elif tracking == 'D': N_hedge = t * w_hedge

		k_t = np.argmin(N_hat) if np.any(N_hat <= np.sqrt(t) - K/2) else \
			np.argmax(N_hedge - N_hat)
		Σ_hat[k_t] += np.random.normal(μ[k_t])
		N_hat[k_t] += 1
		μ_hat = Σ_hat / N_hat[:, np.newaxis]
		cloud_hat = creator(μ_hat)
		t += 1

		Z_t, _ = cloud_hat.get_cost(N_hat/t)
		Z_t = t*Z_t - np.log(np.log(1+t))
		stop = Z_t >= lδ
		if not silent and (t == 1 or stop or t % 20 == 0):
			print(f'{t}/{T}: {Z_t:.3f} { ">" if stop else "≤" } {lδ:.3f}')
		if stop: break
	correct_answer = creator(μ).front[0]
	answer = cloud_hat.front[0]
	correct = np.shape(answer) == np.shape(correct_answer) and np.all(answer == correct_answer)
	return answer, correct, t


if __name__ == '__main__':
	K, d, lδ = 20, 3, np.log(1/1e-2)
	μ = draw_points((K,d))
	track_and_stop(μ, lδ)
