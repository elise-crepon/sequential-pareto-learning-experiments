'''
This programs implement a generic algorithm to identify the Pareto front in a
Multi-Armed Bandits setting.
'''
import numpy as np

from pareto_generic import GPC


def update_graph(G, j, Δ):
	if np.any(Δ + G[:,j] < 0): return False
	G[j,:] = np.min(Δ[:, np.newaxis] + G, axis=0)
	G[...] = np.minimum(G, G[:,j,np.newaxis] + G[np.newaxis,j,:])
	return True


class PCnd(GPC):
	def __init__(self, μ):
		super().__init__(μ)
		self.cells = None

	def get_cells(self):
		if self.cells is not None:
			return
		G, v = np.inf * np.ones((self.d,self.d)), \
			-np.ones(self.p, dtype=int)
		np.fill_diagonal(G, 0)
		self.cells = []
		self.__get_cells(G, v, 0)

	def __get_cells(self, G, v, k):
		if k >= self.p:
			self.cells.append(v.copy())
			return
		point = self.μ[self.front][k]
		for j in range(self.d):
			G_, Δ = np.copy(G), np.minimum(G[j], point - point[j])
			if update_graph(G, j, Δ):
				v[k] = j
				self.__get_cells(G, v, k+1)
			G = G_

	def get_addk_cost(self, μp, wp, μ0, w0):
		g_addk, grad_addkp, grad_addk0 = np.inf, np.zeros(np.shape(wp)), 0.
		self.get_cells()
		for cell in self.cells:
			g_cell, grad_cellp, grad_cell0 = \
				self.get_cell_cost(cell, μp, wp, μ0, w0)
			if g_cell <= g_addk:
				g_addk, grad_addkp, grad_addk0 = \
					g_cell, grad_cellp, grad_cell0
		return g_addk, grad_addkp, grad_addk0

	def get_cell_cost(self, cell, μp, wp, μ0, w0):
		λ0 = np.zeros(self.d)
		g_cell, gradp, grad0 = 0., np.zeros(np.shape(wp)), 0.
		for j in range(self.d):
			cellj, μj, wj = \
				cell[self.sorts[j]], μp[self.sorts[j],j], wp[self.sorts[j]]
			sj, s = np.where(cellj == j), np.where(cell == j)
			μj, wj, μ0j = μj[sj], wj[sj], μ0[j]
			if np.size(μj, axis=0) == 0: continue

			μj_f, wj_f = \
				np.flip(np.append(μj, 0)), np.flip(np.append(wj, 0))
			μj_f = np.flip(μ0j*w0 + np.cumsum(μj_f * wj_f))
			wj_f = np.flip(w0+ np.cumsum(wj_f))
			xj = μj_f/wj_f
			kj = np.argwhere(np.logical_and(
				np.insert(μj, 0, -np.inf) <= xj, xj <= np.append(μj, np.inf)
			))[0,0]
			λ0[j] = xj[kj]

			Δ_g_cell = 1/2 * w0 *(λ0[j] - μ0[j])**2 \
				+ 1/2 * np.sum((wj * np.maximum(0, μj - λ0[j])**2))
			g_cell += Δ_g_cell
			assert(np.abs(w0 * (λ0[j] - μ0[j]) -\
				np.sum(wj * np.maximum(μj - λ0[j], 0))) <= 1e-12
			)

			gradpj = 1/2 * np.maximum(0, μj - λ0[j] )**2
			grad_swap = np.zeros(np.shape(wp))
			grad_swap[sj] = gradpj
			grad_swap[self.sorts[j]] = grad_swap.copy()
			gradp[s] = grad_swap[s]
			grad0 += 1/2 * (λ0[j] - μ0[j])**2

		assert(np.abs(g_cell -  np.sum(wp *gradp) - w0*grad0) <= 1e-12)
		return g_cell, gradp, grad0
