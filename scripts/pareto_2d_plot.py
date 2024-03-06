import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 36})
plt.rcParams.update({'figure.figsize': (30, 10)})
scaling = 4
params = [
  'lines.linewidth', 'axes.linewidth',
  'xtick.minor.size', 'xtick.major.size', 'xtick.minor.width', 'xtick.major.width',
  'ytick.minor.size', 'ytick.major.size', 'ytick.minor.width', 'ytick.major.width',
]
for param in params: plt.rcParams[param] *= scaling
plt.rcParams.update({'xtick.minor.bottom': False})
plt.rcParams.update({'ytick.minor.left': False})
plt.rcParams.update({'lines.dash_capstyle': 'round'})
plt.rcParams.update({'lines.solid_capstyle': 'round'})


def get_add_cost_(μ0, w0, μp, wp):
	gradwp = np.zeros(np.shape(wp))

	idx_dom = np.where(np.all(μp - μ0[None,:] > 0, axis=1))
	β = 1/np.sqrt(2)
	b, n = np.array([β,-β]), np.array([1,1])
	[sd, td] = np.linalg.inv(np.array([b,n]).T) @ (μp[idx_dom] - μ0[None,:]).T
	wd = wp[idx_dom]
	γ = np.vstack(((-b[0]) * sd - td , (-b[1]) * sd - td)).T

	# k_l, k_m, k_d track wether we have passed the line going from k in the
	# left diagonal or down (we pass through those lines at most once)
	k, k_l, k_m, k_d = np.size(sd, axis=0), 0, 0, 0

	σ1, σ2, σ3, σ4, σ5 = 0, 0, 0, 0, 0
	# σ1 = Σ w_k(s) | σ2 = Σ w_k(s) -b_{j_k(s)} | σ3 = Σ w_k(s) γ_k(s)
	# σ4 = Σ w_k(s) -b_{j_k}(s) γ_k(s) | σ5 = Σ w_k(s) γ_k(s)²

	s0 = -np.inf
	plots = np.zeros((3*np.size(sd, axis=0) + 2, 2))
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
		s0p,s1p = s0,s1
		if s0p == -np.inf: s0p = s1p-2
		if s1p == np.inf: s1p = s0p+2
		if s0 == -np.inf: plots[0] = [s0p, α*(s0p - sc) + tc]
		plots[k_d+k_m+k_l+1] = [s1p, α*(s1p-sc) + tc]

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

	for i in range(np.size(td, axis=0)):
		plt.axvline(x=sd[i], color='green', ls=':')
	for i in range(np.size(td, axis=0)):
		x = np.array([
			sd[i]+td[i]/b[0], sd[i], sd[i]+td[i]/b[1]])
		y = -b[0] *np.abs(sd[i] - x) + td[i]
		line, = plt.plot(x, y, color='red', linestyle='--')
	line.set_label('t₁,t₂,t₃')
	plt.plot(plots[:,0], plots[:,1], label='t*', lw=8.)

	plt.xticks(list(sd), ['s₁', 's₂', 's₃'] )
	plt.yticks([0,1,2,3])
	plt.legend()
	plt.show()


if __name__ == '__main__':
	K = 10
	μ = np.array([[1,6], [2,3], [4,1.5]], dtype=float)
	p = 1/6
	get_add_cost_(np.zeros(2), p, μ, (1-p)/3 * np.ones(3,))

