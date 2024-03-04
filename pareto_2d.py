'''
This program implements the 2d specific algorithm to learn the Pareto front in a
Multi-Armed Bandits setting.
'''
import numpy as np

from pareto_generic import GPC


class PC2d(GPC):
	def __init__(self, μ):
		super().__init__(μ)
		assert(self.d == 2)

	def get_addk_cost(self, μp, wp, μ0, w0):
		gradwp, gradwp_sort = np.zeros(np.shape(wp)), np.zeros(np.shape(wp))

		idx_sort = self.sorts[0]
		μp_sort, wp_sort = μp[idx_sort], wp[idx_sort]
		idx_dom = np.where(np.all(μp_sort - μ0[None,:] > 0, axis=1))
		β = 1/np.sqrt(2)
		b, n = np.array([β,-β]), np.array([1,1])
		[sd, td] = np.linalg.inv(np.array([b,n]).T) @ (μp_sort[idx_dom] - μ0[None,:]).T
		wd = wp_sort[idx_dom]
		γ = np.vstack(((-b[0]) * sd - td , (-b[1]) * sd - td)).T

		# k_l, k_m, k_d track wether we have passed the line going from k in the
		# left diagonal or down (we pass through those lines at most once)
		k, k_l, k_m, k_d = np.size(sd, axis=0), 0, 0, 0

		σ1, σ2, σ3, σ4, σ5 = 0, 0, 0, 0, 0
		# σ1 = Σ w_k(s) | σ2 = Σ w_k(s) -b_{j_k(s)} | σ3 = Σ w_k(s) γ_k(s)
		# σ4 = Σ w_k(s) -b_{j_k}(s) γ_k(s) | σ5 = Σ w_k(s) γ_k(s)²

		s0 = -np.inf
		s_star, t_star, g_star = -np.inf, np.inf, np.inf
		while True:
			assert(k_d <= k_m <= k_l)
			α = σ2/(2*w0 + σ1)
			[sc, tc] = -np.linalg.inv([[w0+1/2*σ1, -σ2], [-σ2, 2*w0+σ1]]) @ [-σ4, σ3]

			crosses = np.array([
				np.inf,
				(-γ[k_l][1] + α*sc - tc)/(α - (-b[1])) \
					if k_l < k else np.inf, # crossing an increasing line
				sd[k_m] if k_m < k else np.inf, # crossing a vertical line
				(-γ[k_d][0] + α*sc - tc)/(α - (-b[0])) \
					if k_d < k else np.inf, # crossing a decreasing line
			])
			cross, s1 = np.argmin(crosses), np.min(crosses)

			s_clip = np.clip(sc, s0, s1)
			sc, tc = s_clip, α*(s_clip-sc) + tc
			gc = 1/2 * ((w0 + 1/2*σ1)*(sc**2 + 2*tc**2) + σ5 - 2*sc*tc*σ2 - 2*sc*σ4 + 2*tc*σ3)

			if gc < g_star:
				s_star, t_star, g_star = sc, tc, gc

			if cross == 0: break
			elif cross == 1:
				σ1 += wd[k_l]
				σ2 += wd[k_l] * (-b[1])
				σ3 += wd[k_l] * γ[k_l,1]
				σ4 += wd[k_l] * (-b[1]) * γ[k_l,1]
				σ5 += wd[k_l] * γ[k_l,1]**2
				k_l += 1
			elif cross == 2:
				σ2 += wd[k_m] * ((-b[0]) - (-b[1]))
				σ3 += wd[k_m] * (γ[k_m,0] - γ[k_m,1])
				σ4 += wd[k_m] * ((-b[0]) * γ[k_m,0] - (-b[1]) * γ[k_m,1])
				σ5 += wd[k_m] * (γ[k_m,0]**2 - γ[k_m,1]**2)
				k_m += 1
			elif cross == 3:
				σ1 -= wd[k_d]
				σ2 -= wd[k_d] * (-b[0])
				σ3 -= wd[k_d] * γ[k_d,0]
				σ4 -= wd[k_d] * (-b[0]) * γ[k_d,0]
				σ5 -= wd[k_d] * γ[k_d,0]**2
				k_d += 1
			t1 = α * (s1 - sc) + tc
			s0 = s1

		gradw0 = 1/2 * (s_star**2 + 2*t_star**2)
		tds = td - β*np.abs(s_star-sd)
		gradwp_sort[idx_dom] = 1/2*np.where(tds > t_star, tds - t_star, 0)**2
		gradwp[idx_sort] = gradwp_sort

		assert(np.abs(g_star - np.dot(wp, gradwp) - w0 * gradw0) <= 1e-12)
		return g_star, gradwp, gradw0

	def get_rm_cost(self, μp, wp):
		g_rm, grad_rm = np.inf, np.zeros(np.shape(wp))
		if np.size(wp, axis=0) == 1: return g_rm, grad_rm

		idxs = self.sorts[0]
		grad_rms = np.zeros(np.shape(wp))
		μps, wps = μp[idxs], wp[idxs]
		Δ = np.diff(μps, axis=0)
		Δ = np.minimum(
			np.sum(np.maximum(0,Δ)**2, axis=-1),
			np.sum(np.minimum(0,Δ)**2, axis=-1),
		)
		g_rm = 1/2 * 1/(1/wps[:-1] + 1/wps[1:]) * Δ
		g_rm, k_rm = np.min(g_rm), np.argmin(g_rm)

		grad_rms[k_rm] = 1/2/(1 + wps[k_rm]/wps[k_rm+1])**2 * Δ[k_rm]
		grad_rms[k_rm+1] = 1/2/(1 + wps[k_rm+1]/wps[k_rm])**2 * Δ[k_rm]

		grad_rm[idxs] = grad_rms
		assert(np.abs(g_rm - np.dot(grad_rm, wp)) <= 1e-12)
		return np.dot(grad_rm, wp), grad_rm
