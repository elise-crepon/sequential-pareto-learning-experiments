import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 24})
plt.rcParams['lines.linewidth'] = 3.0
plt.rcParams['lines.markersize'] = 18


if __name__ == '__main__':
	data = np.array([
		[6.18e-4, 9.79e-4, 1.88e-3, 4.20e-3, 1.09e-2, 3.44e-2, 1.17e-1 ], # dimension 2
		[7.36e-4, 1.87e-3, 5.64e-3, 2.19e-2, 1.03e-1, 5.77e-1, 3.62e+0 ], # dimension 3
		[9.47e-4, 3.13e-3, 1.42e-2, 8.84e-2, 7.18e-1, 6.99e+0, 8.17e+1 ], # dimension 4
		[1.17e-3, 5.03e-3, 2.97e-2, 2.93e-1, 3.91e+0, 6.71e+1, 1.42e+3 ], # dimension 5
	])
	p = np.array([1, 2, 4, 8, 16, 32, 64])
	data = data[:,2:]
	p = p[2:]
	ds = [2, 3, 4, 5]
	styles = ['-', ':', '--', '-.']
	markers = ['x', 'o', '*', 'D']
	for i,d in enumerate(ds):
		plt.plot(p, data[i], label=f'd = {d}', linestyle=styles[i], marker=markers[i])
	plt.xlabel('p')
	plt.xscale('log')
	plt.xticks(p, p)
	plt.ylabel('time (s)')
	plt.yscale('log')
	plt.legend(loc='upper left')
	plt.show()

#d=2 p=1: tn:6.18e-04 (±8.93e-04)
#	d=3 p=1: tn:7.36e-04 (±5.66e-05)
#		d=4 p=1: tn:9.47e-04 (±4.20e-05)
#			d=5 p=1: tn:1.17e-03 (±6.62e-05)
#d=2 p=2: tn:9.79e-04 (±6.47e-05)
#	d=3 p=2: tn:1.87e-03 (±1.05e-04)
#		d=4 p=2: tn:3.13e-03 (±1.80e-04)
#			d=5 p=2: tn:5.03e-03 (±1.70e-04)
#d=2 p=4: tn:1.88e-03 (±1.22e-04)
#	d=3 p=4: tn:5.64e-03 (±1.44e-04)
#		d=4 p=4: tn:1.42e-02 (±7.90e-04)
#			d=5 p=4: tn:2.97e-02 (±4.90e-04)
#d=2 p=8: tn:4.20e-03 (±2.12e-04)
#	d=3 p=8: tn:2.19e-02 (±5.26e-04)
#		d=4 p=8: tn:8.84e-02 (±9.79e-04)
#			d=5 p=8: tn:2.93e-01 (±6.90e-03)
#d=2 p=16: tn:1.09e-02 (±2.30e-04)
#	d=3 p=16: tn:1.03e-01 (±1.38e-03)
#		d=4 p=16: tn:7.18e-01 (±1.62e-02)
#			d=5 p=16: tn:3.91e+00 (±5.51e-02)
#d=2 p=32: tn:3.44e-02 (±1.73e-03)
#	d=3 p=32: tn:5.75e-01 (±1.35e-02)
#		d=4 p=32: tn:6.99e+00 (±6.88e-02)
#			d=5 p=32: tn:6.71e+01 (±4.01e-01)
#d=2 p=64: tn:1.17e-01 (±2.42e-03)
#	d=3 p=64: tn:3.62e+00 (±4.52e-02)
#		d=4 p=64: tn:8.17e+01 (±5.30e-01)
#			d=5 p=64:  tn:1413.82s
