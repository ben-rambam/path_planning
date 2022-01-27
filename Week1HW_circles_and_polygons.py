#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:56:04 2022

@author: jonathan
"""

import numpy as np
import matplotlib.pyplot as plt

# %% Problem 1
"""
Generate a list of N random floating point numbers (x,y,r), r  > 0. Can you
plot N circles with center given by (x,y) and radius r? N does not need to be
more than 5.   Can you shade the interior?  Could you determine the plot region
on your own?
"""

n = 5

circs = np.random.uniform([-5, -5, 0.2], [5, 5, 1], (5, 3))

fig, [ax1, ax2] = plt.subplots(2, 1)
for dims in circs:
    circle = plt.Circle((dims[0], dims[1]), dims[2])
    ax1.add_patch(circle)

ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.axis('equal')


# %% Problem 2
"""
Can you plot N random convex polygons?  A random convex polygon would be a 
convex polygon with random vertices.   You can stick with 4-8 vertices.  Can 
you shade the interior?  For this problem try something easy. Pick N random 
points on a circle.  Sort according to angle.  Use those as vertices.   [We 
can make this more robust if we wanted.  If you generate a random set of points 
in the plane, the convex hull is a convex polygon.  However, the number of 
vertices is probably less than the number of generated points and so we need to 
modify our data set.]
"""
for dims in circs:
    x = dims[0]
    y = dims[1]
    r = dims[2]

    n = 6

    ts = np.sort(np.random.uniform(0, 2*np.pi, n))
    rs = np.random.uniform(r*0.9, r*1.1, n)

    pts = np.array([rs*np.cos(ts) + x, rs*np.sin(ts) + y]).T

    poly = plt.Polygon(pts)
    ax2.add_patch(poly)
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.axis('equal')
