'''
This program implements common tools between the generic algorithm and the 2d
specific one.
'''
import numpy as np


class GPC:
	def __init__(self, μ):
		self.μ = μ
		self.K, self.d = np.shape(μ)
		self.get_pareto_set()

	def get_pareto_set(self):
		Δ = self.μ[None,:,:] - self.μ[:,None,:]
		Δ = np.all(Δ >= 0, axis=2)
		np.fill_diagonal(Δ, False)
		front = np.all(np.logical_not(Δ), axis=1)
		self.front, self.back = np.where(front), \
			np.where(np.logical_not(front))
		self.sorts = [
			np.argsort(self.μ[front, i]) for i in range(self.d) ]
		self.p = np.size(self.μ[self.front], axis=0)

	def get_rm_cost(self, μp, wp):
		grad_rm = np.zeros(np.shape(wp))
		if np.size(wp, axis=0) == 1: return np.inf, grad_rm

		Δ = μp[:, np.newaxis] - μp[np.newaxis, :]
		Δ = np.sum(np.where (Δ > 0, Δ**2, 0), axis=-1)
		C = 1/2 * 1/(1/wp[:, np.newaxis] + 1/wp[np.newaxis,:]) * Δ
		np.fill_diagonal(C, np.inf)
		g_rm, (k0, k1) = np.min(C, axis=None), \
			np.unravel_index(np.argmin(C, axis=None), np.shape(C))
		grad_rm[k0] = 1/2 * (1/(1 + wp[k0]/wp[k1]))**2 * Δ[k0][k1]
		grad_rm[k1] = 1/2 * (1/(1 + wp[k1]/wp[k0]))**2 * Δ[k0][k1]

		assert(np.abs(g_rm - np.sum(wp*grad_rm)) <= 1e-12)
		return g_rm, grad_rm

	def get_add_cost(self, μp, wp, μ0, w0):
		g_add, grad_addp, grad_add0 = \
			np.inf, np.zeros(np.shape(wp)), np.zeros(np.shape(w0))
		if np.size(w0, axis=0) == 0:
			return g_add, grad_addp, grad_add0
		for k in range(np.size(w0, axis=0)):
			gk, gradp, grad0 = \
				self.get_addk_cost(μp, wp, μ0[k], w0[k])
			if gk <= g_add:
				g_add, grad_addp = gk, gradp
				idxk, grad_addk = k, grad0
		grad_add0[idxk] = grad_addk
		return g_add, grad_addp, grad_add0

	def get_addk_cost(self, μp, wp, μ0, w0):
		return np.inf, np.zeros(np.shape(wp))

	def get_cost(self, w):
		μp, wp = self.μ[self.front], w[self.front]
		μ0, w0 = self.μ[self.back], w[self.back]

		g_add, grad_addp, grad_add0 = self.get_add_cost(μp, wp, μ0, w0)
		g_rm, grad_rmp = self.get_rm_cost(μp, wp)

		if g_rm <= g_add:
			g = g_rm
			gradp, grad0 = grad_rmp, np.zeros(np.shape(w0))
		else:
			g = g_add
			gradp, grad0 = grad_addp, grad_add0

		grad = np.zeros(np.shape(w))
		grad[self.front] = gradp
		grad[self.back] = grad0

		assert(np.abs(g - np.sum(w*grad)) <= 1e-12)
		return g, grad
