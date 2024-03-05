'''
This program was used to generate Figure 2 in the paper. The boxplot.json file
was obtained as described in the section Experiments of the paper.
'''
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 24})

if __name__ == '__main__':
	with open('boxplot.json') as f:
		results = json.load(f)
	samples = np.array([
		result['sample'] for result in results
	])
	bins = np.histogram_bin_edges(samples, bins=50, range=(0,np.quantile(samples,0.99)))
	plt.hist(samples, bins=bins, density=True, edgecolor='black', linewidth=1.4, fc=(0,1,0,0.5), label='τ_δ')
	plt.axvline(x=4845, color='purple', linestyle='--', linewidth=8, label='log(1/δ) T*(θ)')
	plt.axvline(x=np.average(samples), color='red', linewidth=8, label='E_θ(τ_δ)')
	plt.legend(loc='upper right')
	plt.show()
	print(np.average(samples))
