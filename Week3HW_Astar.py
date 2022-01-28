#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:22:03 2022

@author: jonathan
"""
import numpy as np
import matplotlib.pyplot as plt


def make_poly(x, y, r, n):
    ts = np.sort(np.random.uniform(0, 2*np.pi, n))
    rs = np.random.uniform(r*0.9, r*1.1, n)

    pts = np.array([rs*np.cos(ts) + x, rs*np.sin(ts) + y]).T

    poly = plt.Polygon(pts)
    return poly


def generate_random_obstacles(n, obs_type, start_pt, goal_pt):
    sigma = 2
    obstacles = []
    for i in range(n):
        x = np.random.normal(0, sigma)
        y = np.random.normal(0, sigma)
        r = np.random.uniform(0.5, 1)
        if obs_type == "circles":
            patch = plt.Circle([x, y], r)
        elif obs_type == "polygons":
            patch = make_poly(x, y, r, 6)
        while patch.contains_point(start_pt) or patch.contains_point(goal_pt):
            print("retry making obstacle")
            x = np.random.normal(0, sigma)
            y = np.random.normal(0, sigma)
            r = np.random.uniform(0.5, 1)
            if obs_type == "circles":
                patch = plt.Circle([x, y], r)
            elif obs_type == "polygons":
                patch = make_poly(x, y, r, 6)
        obstacles.append(patch)
    return obstacles


def convert_obstacles_to_image(obstacles, img_shape):
    img = np.ones(img_shape)
    xmin = -5
    xmax = 5
    xs = np.linspace(xmin, xmax, img_shape[1])
    ymin = xmin*img_shape[0]/img_shape[1]
    ymax = xmax*img_shape[0]/img_shape[1]
    ys = np.linspace(ymin, ymax, img_shape[0])

    for obs in obstacles:
        extent = obs.get_window_extent()
        print(extent)
        imin = 0 if extent.x0 < xmin else np.argwhere(xs > extent.x0)[0][0]
        imax = img_shape[1] - \
            1 if extent.x1 > xmax else np.argwhere(xs > extent.x1)[0][0]
        jmin = 0 if extent.y0 < ymin else np.argwhere(ys > extent.y0)[0][0]
        jmax = img_shape[0] - \
            1 if extent.y1 > ymax else np.argwhere(ys > extent.y1)[0][0]
        for i in range(imin, imax):
            for j in range(jmin, jmax):
                if obs.contains_point([xs[i], ys[j]]):
                    img[j, i] = 0
    return img


obstacles = generate_random_obstacles(10, "circles", [-6, 3], [6, 3])
img = convert_obstacles_to_image(obstacles, (480, 640))
plt.imshow(img)


class Node():
    def __init__(self):
        self.neighbors = []
    pass


class Graph():
    def __init__(self):
        self.nodes = {}
    pass


def convert_img_to_graph(img):
    graph = Graph()
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            node = Node()
            node.x = i
            node.y = j
            for pair in [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]:
                a = i+pair[0]
                b = j+pair[1]
                if a >= 0 and a < img.shape[1] and b >= 0 and b < img.shape[0] and img[b, a]*img[j, i] == 1:
                    node.neighbors.append(np.linalg.norm(pair))
                else:
                    node.neighbors.append(10000)
            graph.nodes[(i, j)] = node
    return graph


graph = convert_img_to_graph(img)

