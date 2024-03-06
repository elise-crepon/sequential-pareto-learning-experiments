'''
This program was used to generate Figure 2 in the paper. The boxplot.json file
was obtained as described in the section Experiments of the paper.
'''
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 45})
plt.rcParams.update({'text.latex.preamble': r'\usepackage{amsfonts}'})
plt.rcParams.update({'figure.figsize': (30, 10)})

if __name__ == '__main__':
	with open('boxplot.json') as f:
		results = json.load(f)
	samples = np.array([
		result['sample'] for result in results
	])
	bins = np.histogram_bin_edges(samples, bins=50, range=(0,np.quantile(samples,0.99)))
	plt.hist(samples, bins=bins, density=True, edgecolor='black', linewidth=1.4, fc=(0,1,0,0.5), label=r'$\tau_\delta$')
	plt.axvline(x=4845, color='purple', linestyle='--', linewidth=8, label=r'$\log\left(\frac{1}{\delta}\right) T^*(\theta)$')
	plt.axvline(x=np.average(samples), color='red', linewidth=8, label=r'$\mathbb{E}_\theta(\tau_\delta)$')
	plt.legend(loc='upper right')
	plt.savefig('../../writeup/figs/graph.pdf')
