#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:22:03 2022

@author: jonathan
"""
import numpy as np
import matplotlib.pyplot as plt
import heapq
import cv2


# %%


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


def convert_obstacles_to_image(obstacles, img_shape, start_pt, goal_pt):
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
        imin = int((min(max(xmin, extent.x0), xmax)-xmin)
                   * img_shape[1]/(xmax-xmin))
        imax = int((min(max(xmin, extent.x1), xmax)-xmin)
                   * img_shape[1]/(xmax-xmin))
        jmin = int((min(max(ymin, extent.y0), ymax)-ymin)
                   * img_shape[0]/(ymax-ymin))
        jmax = int((min(max(ymin, extent.y1), ymax)-ymin)
                   * img_shape[0]/(ymax-ymin))
        for i in range(imin, imax):
            for j in range(jmin, jmax):
                if obs.contains_point([xs[i], ys[j]]):
                    img[j, i] = 0
    A = np.array([[img_shape[1]/(xmax-xmin), 0],
                  [0, img_shape[0]/(ymax-ymin)]])
    xscale = img_shape[1]/(xmax-xmin)
    yscale = img_shape[0]/(ymax-ymin)
    new_start = np.array(
        [(start_pt[0]-xmin)*xscale, (start_pt[1]-ymin)*yscale], dtype='int')
    new_goal = np.array(
        [(goal_pt[0]-xmin)*xscale, (goal_pt[1]-ymin)*yscale], dtype='int')
    return img, new_start, new_goal


def createMap(num_obstacles, size):
    start_pt = np.array([-3, 3])
    goal_pt = np.array([3, -3])
    start_pt = np.random.uniform([-5, -5], [0, 5], 2)
    goal_pt = np.random.uniform([0, -5], [5, 5], 2)
    obstacles = generate_random_obstacles(
        num_obstacles, "circles", start_pt, goal_pt)
    img, img_start_pt, img_goal_pt = convert_obstacles_to_image(
        obstacles, (size, size), start_pt, goal_pt)
    return img, img_start_pt, img_goal_pt
# plt.imshow(img)
# plt.plot(img_start_pt[0], img_start_pt[1], 'gs')
# plt.plot(img_goal_pt[0], img_goal_pt[1], 'r*')


# %%

def loadMap(filename, cells):
    img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)/255.0
    img = img[1:-1, 1:-1]
    img = cv2.erode(img, kernel=np.ones(
        (3, 3), dtype=np.float64), iterations=3)
    img = cv2.resize(img, (cells*4, cells*4))
    img_start_pt = np.random.randint([0, 0], [img.shape[1], img.shape[0]], 2)
    while(img[img_start_pt[1], img_start_pt[0]] < 0.5):
        img_start_pt = np.random.randint(
            [0, 0], [img.shape[1], img.shape[0]], 2)
    img_goal_pt = np.random.randint([0, 0], [img.shape[1], img.shape[0]], 2)
    while (img[img_goal_pt[1], img_goal_pt[0]] < 0.5):
        img_goal_pt = np.random.randint(
            [0, 0], [img.shape[1], img.shape[0]], 2)

    return img, img_start_pt, img_goal_pt


# img, img_start_pt, img_goal_pt = createMap(10, 160)
img, img_start_pt, img_goal_pt = loadMap("20x20_orthogonal_maze.png", 20)
plt.imshow(img)

# %%


class Node():
    def __init__(self, hfunc):
        self.neighbors = []
        self.g = np.Inf
        self.hfunc = hfunc

    def __lt__(self, other):
        lt = False
        f1 = self.g+self.hfunc(np.array([self.x, self.y]))
        f2 = other.g+self.hfunc(np.array([other.x, other.y]))
        if np.abs(f1-f2) < 0.001:
            lt = self.g > other.g
        else:
            lt = f1 < f2
        return lt

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class Graph():
    def __init__(self):
        self.nodes = {}
    pass


def h_l2_dist(node):
    return np.linalg.norm(node-img_goal_pt)


def grid_dist(p1, p2):
    dx = np.abs(p1[0]-p2[0])
    dy = np.abs(p1[1]-p2[1])
    if dx > dy:
        return (dx-dy)+np.sqrt(2)*dy
    else:
        return (dy-dx)+np.sqrt(2)*dx


def h_grid_dist(node):
    return grid_dist(node, img_goal_pt)


def h_hyperbolic_1(node):
    return grid_dist(node, img_goal_pt)-grid_dist(node, img_start_pt)


def h_hyperbolic_2(node):

    fake_pt = img_start_pt - (img_goal_pt - img_start_pt)
    return grid_dist(node, img_goal_pt) - grid_dist(node, fake_pt)


def h_hyperbolic_3(node):

    fake_pt = img_start_pt - 2*(img_goal_pt - img_start_pt)
    return grid_dist(node, img_goal_pt) - grid_dist(node, fake_pt)


def h_parallel_goal_dist(node):
    dir_vec = img_goal_pt-img_start_pt
    unit_vec = dir_vec/np.linalg.norm(dir_vec)

    pos_vec = img_goal_pt-node
    return np.dot(pos_vec, unit_vec)


def convert_img_to_graph(img, h):
    graph = Graph()
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            node = Node(h)
            node.x = i
            node.y = j
            for pair in [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]:
                a = i+pair[0]
                b = j+pair[1]
                if a >= 0 and a < img.shape[1] and b >= 0 and b < img.shape[0] and img[b, a]*img[j, i] == 1:
                    node.neighbors.append(np.linalg.norm(pair))
                else:
                    node.neighbors.append(np.Inf)
            graph.nodes[(i, j)] = node
    return graph


def a_star(graph, start, goal):
    start_node = graph.nodes[start[0], start[1]]
    start_node.g = 0

    goal_node = graph.nodes[goal[0], goal[1]]

    start_node.goal = goal
    Q = [start_node]
    heapq.heapify(Q)
    curr_node = heapq.heappop(Q)
    searched = []
    while curr_node != goal_node:
        searched.append([curr_node.x, curr_node.y])
        for pair, weight in zip([[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]], curr_node.neighbors):
            a = curr_node.x + pair[0]
            b = curr_node.y + pair[1]
            try:
                other_node = graph.nodes[a, b]
                if curr_node.g + weight < other_node.g:
                    other_node.g = curr_node.g + weight
                    other_node.goal = goal
                    other_node.prev = curr_node
                    heapq.heappush(Q, other_node)
            except:
                pass
        curr_node = heapq.heappop(Q)

    path = [[curr_node.x, curr_node.y]]
    path_length = 0
    while curr_node != start_node:
        path.insert(0, [curr_node.prev.x, curr_node.prev.y])
        path_length += np.linalg.norm([curr_node.x -
                                      curr_node.prev.x, curr_node.y-curr_node.prev.y])
        curr_node = curr_node.prev

    return graph, np.array(path), np.array(searched), path_length


hs = [h_grid_dist, h_hyperbolic_1, h_hyperbolic_2, h_hyperbolic_3]
hnames = ['grid circle', 'hyperbolic 1', 'hyperbolic 2', 'hyperbolic 3']
fig, axes = plt.subplots(1, len(hs))
for h, name, hnum in zip(hs, hnames, range(len(hs))):

    graph = convert_img_to_graph(img, h)
    new_graph, path, searched, path_length = a_star(
        graph, img_start_pt, img_goal_pt)

    ax = axes[hnum]
    ax.imshow(img)
    ax.set_xlabel(name + "\nlength: " + str(path_length) +
                  " \nsearched: " + str(len(searched)))
    ax.plot(img_start_pt[0], img_start_pt[1], 'gs')
    ax.plot(img_goal_pt[0], img_goal_pt[1], 'r*')
    ax.plot(searched[:, 0], searched[:, 1], '.')
    ax.plot(path[:, 0], path[:, 1])
