'''
This program provide three utilities one made to draw a random point cloud from a
uniform distribution, the second one that add new points in the Pareto front
of an existing point cloud (only 2d) and the last one that draws a point cloud
on the all-positive quadrant part of a sphere of a given radius.
'''
import numpy as np
import sys


def draw_points(shape, width=10.):
	return np.random.uniform(0, width, size=shape)


def draw_only_pareto(μ, width=10.):
	N,K,d = np.shape(μ)
	assert(d == 2)

	rμ = μ[...,[0]]
	μs = np.take_along_axis(μ, np.argsort(μ[...,[0]], axis=1), axis=1)
	z,o = np.zeros((N,1)), width * np.ones((N,1))
	μl0,μl1,μr1,μr0,μl01,μr01 = ( np.concatenate(p, axis=1) \
		for p in [(z,μs[...,0]), (μs[...,0],o), (o,μs[...,1]), (μs[...,1],z),
			(z,μs[...,0],o), (o,μs[...,1],z)]
	)
	d = -np.diff(μl01) * np.diff(μr01)
	d /= np.sum(d, axis=1)[:,np.newaxis]
	d = np.concatenate((np.zeros((N,1)),np.cumsum(d, axis=1)), axis=1)
	rdx = np.random.uniform(size=(N,1))
	idx = np.where(np.logical_and(d[...,:-1] <= rdx , rdx < d[...,1:]))
	root = np.stack((μl0[idx], μr1[idx]), axis=-1)
	sky = np.stack((μl1[idx], μr0[idx]), axis=-1)
	new_points = root + (sky-root) * np.random.uniform(size=(N,2))
	return np.concatenate((μ, new_points[:,np.newaxis,:]), axis=1)


def draw_spherical(shape, width=10.):
    μ = np.abs(np.random.normal(size=shape))
    μ /= np.linalg.norm(μ, axis=-1)[..., np.newaxis] * width
    return μ
