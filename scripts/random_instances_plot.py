import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 32})
plt.rcParams.update({'figure.figsize': (15, 10)})
plt.subplots_adjust(bottom=0.125)
scaling = 5
params = [
  'lines.linewidth', 'axes.linewidth',
  'xtick.minor.size', 'xtick.major.size', 'xtick.minor.width', 'xtick.major.width',
  'ytick.minor.size', 'ytick.major.size', 'ytick.minor.width', 'ytick.major.width',
]
for param in params: plt.rcParams[param] *= scaling
plt.rcParams['lines.markersize'] *= 4
plt.rcParams['lines.markeredgewidth'] *= 3.5
plt.rcParams.update({'xtick.minor.bottom': False})
plt.rcParams.update({'ytick.minor.left': False})


if __name__ == '__main__':
	data = np.array([
		[6.18e-4, 9.79e-4, 1.88e-3, 4.20e-3, 1.09e-2, 3.44e-2, 1.17e-1 ], # dimension 2
		[7.36e-4, 1.87e-3, 5.64e-3, 2.19e-2, 1.03e-1, 5.77e-1, 3.62e+0 ], # dimension 3
		[9.47e-4, 3.13e-3, 1.42e-2, 8.84e-2, 7.18e-1, 6.99e+0, 8.17e+1 ], # dimension 4
		[1.17e-3, 5.03e-3, 2.97e-2, 2.93e-1, 3.91e+0, 6.71e+1, 1.42e+3 ], # dimension 5
	])
	p = np.array([1, 2, 4, 8, 16, 32, 64])
	data, p = data[:,2:], p[2:]
	ds = [2, 3, 4, 5]
	styles = ['-', ':', '--', '-.']
	markers = ['D', 's', 'o', 'p']
	for i,d in enumerate(ds):
		plt.plot(p, data[i], label=f'd = {d}', linestyle=styles[i], marker=markers[i], mfc='w',)

	plt.xlabel('p')
	plt.xscale('log')
	plt.xticks(p, p)
	plt.ylabel('time (s)')
	plt.yscale('log')
	leg = plt.legend(loc='upper left')
	for line in leg.get_lines():
		line.set_linestyle('-')

	plt.savefig('../../writeup/figs/complexityexp.pdf')
