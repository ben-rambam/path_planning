#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 22:22:03 2022

@author: jonathan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import heapq
import cv2
import time

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


def create_map(num_obstacles, obstacle_type, map_size):
    start_pt = np.array([-3, 3])
    goal_pt = np.array([3, -3])
    start_pt = np.random.uniform([-5, -5], [0, 5], 2)
    goal_pt = np.random.uniform([0, -5], [5, 5], 2)
    obstacles = generate_random_obstacles(
        num_obstacles, obstacle_type, start_pt, goal_pt)
    img, img_start_pt, img_goal_pt = convert_obstacles_to_image(
        obstacles, (map_size, map_size), start_pt, goal_pt)
    return img, img_start_pt, img_goal_pt


# %%
"""
mazes generated with https://www.mazegenerator.net/
"""


def load_map(filename, cells):
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

# %%


img, img_start_pt, img_goal_pt = create_map(40, 'polygons', 160)
# img, img_start_pt, img_goal_pt = load_map("40x40_orthogonal_maze.png", 40)
print("img_start_pt", img_start_pt)
print("img_goal_pt", img_goal_pt)

# %%


class Node():
    def __init__(self, hfunc):
        self.neighbors = {}
        self.g = np.Inf
        self.hfunc = hfunc

    def __lt__(self, other):
        lt = False
        f1 = self.g+self.hfunc(np.array([self.x, self.y]))
        f2 = other.g+self.hfunc(np.array([other.x, other.y]))
        # if np.abs(f1-f2) < 0.001:
        #     lt = self.g > other.g
        # else:
        #     lt = f1 < f2
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


def h_sq_dist(node):
    return (node[0] - img_goal_pt[0])**2 + (node[1]-img_goal_pt[1])**2


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


pairs = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
pairs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def convert_img_to_graph(img, h):
    graph = Graph()
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            node = Node(h)
            node.x = i
            node.y = j
            for pair in pairs:
                a = i+pair[0]
                b = j+pair[1]
                if a >= 0 and a < img.shape[1] and b >= 0 and b < img.shape[0] and img[b, a]*img[j, i] == 1:
                    node.neighbors[pair] = np.linalg.norm(pair)
                else:
                    pass
                    # node.neighbors.append(np.Inf)
            graph.nodes[(i, j)] = node
    return graph


def retrace_path(curr_node, start_node):
    path = [[curr_node.x, curr_node.y]]
    path_length = 0
    while curr_node != start_node:
        path.insert(0, [curr_node.prev.x, curr_node.prev.y])
        path_length += np.linalg.norm([curr_node.x -
                                      curr_node.prev.x, curr_node.y-curr_node.prev.y])
        curr_node = curr_node.prev
    return np.array(path), path_length


def a_star(graph, start, goal):
    start_node = graph.nodes[start[0], start[1]]
    start_node.g = 0

    goal_node = graph.nodes[goal[0], goal[1]]

    start_node.goal = goal
    Q = [start_node]
    heapq.heapify(Q)
    searched = []
    count = 2
    while len(Q) > 0:
        curr_node = heapq.heappop(Q)
        # if img_cp[curr_node.y, curr_node.x] > 1:
        #     raise RuntimeError("this shouldn't happen")
        img_cp[curr_node.y, curr_node.x] = count
        count += 1

        if curr_node == goal_node:
            break

        searched.append([curr_node.x, curr_node.y])
        for pair in curr_node.neighbors:
            a = curr_node.x + pair[0]
            b = curr_node.y + pair[1]
            weight = curr_node.neighbors[pair]
            try:
                other_node = graph.nodes[a, b]
                if curr_node.g + weight - other_node.g < -0.00000001:
                    if img_cp[b, a] > 1:
                        raise RuntimeError("This shouldn't happen")
                    other_node.g = curr_node.g + weight
                    other_node.prev = curr_node
                    if other_node not in Q:
                        other_node.goal = goal
                        heapq.heappush(Q, other_node)
            except RuntimeError:
                plt.imshow(img_cp)
                for coords in searched:
                    node = graph.nodes[tuple(coords)]
                    path, pl = retrace_path(node, start_node)
                    plt.plot(path[:, 0], path[:, 1], 'k')
                plt.plot(curr_node.x, curr_node.y, 'gs')
                plt.plot(other_node.x, other_node.y, 'r*')
                plt.plot(start_node.x, start_node.y, 'gs')
                plt.plot(goal_node.x, goal_node.y, 'r*')
                raise RuntimeError("This shouldn't happen")

    path, path_length = retrace_path(curr_node, start_node)

    return graph, np.array(path), np.array(searched), path_length


img_cp = img.copy()
graph = convert_img_to_graph(img_cp, h_l2_dist)
start = img_start_pt
goal = img_goal_pt

start = time.time()
new_graph, path, searched, path_length = a_star(
    graph, img_start_pt, img_goal_pt)
end = time.time()

fig, ax = plt.subplots(1, 1)
im = ax.imshow(img_cp)
ax.set_xlabel("\nlength: {} \nsearched: {} \ntime: {}".format(
    path_length, len(searched), end-start))

ax.plot(img_start_pt[0], img_start_pt[1], 'gs')
ax.plot(img_goal_pt[0], img_goal_pt[1], 'r*')
# ax.plot(searched[:, 0], searched[:, 1], '.')
ax.plot(path[:, 0], path[:, 1])
plt.colorbar(im)


def a_star_generator():
    start_node = graph.nodes[start[0], start[1]]
    start_node.g = 0

    goal_node = graph.nodes[goal[0], goal[1]]

    start_node.goal = goal
    Q = [start_node]
    heapq.heapify(Q)
    searched = []
    while len(Q) > 0:
        curr_node = heapq.heappop(Q)
        if img_cp[curr_node.y, curr_node.x] > 1:
            raise RuntimeError("this shouldn't happen")
        img_cp[curr_node.y, curr_node.x] += 1

        if curr_node == goal_node:
            break

        searched.append([curr_node.x, curr_node.y])
        for pair in curr_node.neighbors:
            a = curr_node.x + pair[0]
            b = curr_node.y + pair[1]
            weight = curr_node.neighbors[pair]
            try:
                other_node = graph.nodes[a, b]
                if curr_node.g + weight - other_node.g < -0.00000001:
                    if img_cp[b, a] > 1:
                        raise RuntimeError("This shouldn't happen")
                    other_node.g = curr_node.g + weight
                    other_node.prev = curr_node
                    if other_node not in Q:
                        other_node.goal = goal
                        heapq.heappush(Q, other_node)
            except RuntimeError:
                raise RuntimeError("This shouldn't happen")

    path, path_length = retrace_path(curr_node, start_node)

    return graph, np.array(path), np.array(searched), path_length


# def animFunc(data):

#     pass


# animation = mpl.animation.FuncAnimation(
#     fig, animFunc, bug1.update, interval=20)

# hs = [h_l2_dist, h_grid_dist, h_sq_dist,
#       h_hyperbolic_1, h_hyperbolic_2, h_hyperbolic_3]
# hnames = ['circle', 'grid circle', 'squared dist', 'hyperbolic 1',
#           'hyperbolic 2', 'hyperbolic 3']
# hs = [h_l2_dist, h_grid_dist]
# hnames = ['circle', 'grid circle']
# for h, name, hnum in zip(hs, hnames, range(len(hs))):
#     img_cp = img.copy()
#     graph = convert_img_to_graph(img, h)
#     start = time.time()

#     new_graph, path, searched, path_length = a_star(
#         graph, img_start_pt, img_goal_pt)
#     end = time.time()

#     print(name, end-start)
#     fig, ax = plt.subplots(1, 1)

#     # ax = axes[hnum]
#     im = ax.imshow(img_cp)
#     ax.set_xlabel("{} \nlength: {} \nsearched: {} \ntime: {}".format(
#         name, path_length, len(searched), end-start))

#     ax.plot(img_start_pt[0], img_start_pt[1], 'gs')
#     ax.plot(img_goal_pt[0], img_goal_pt[1], 'r*')
#     # ax.plot(searched[:, 0], searched[:, 1], '.')
#     ax.plot(path[:, 0], path[:, 1])
#     plt.colorbar(im)


# %% depth_first search A* broken
def a_star_2(graph, start, goal, max_iter=np.Inf):
    start_node = graph.nodes[start[0], start[1]]
    start_node.g = 0

    goal_node = graph.nodes[goal[0], goal[1]]

    start_node.goal = goal
    # examine neighbors
    # if a neighbor has a heuristic smaller than mine
    #  go to neighbor with smallest heuristic
    # else
    #  remove self from neighbors of prev
    #  go to prev
    searched = []
    curr_node = start_node
    path = [curr_node]
    count = 0
    while curr_node != goal_node and count < max_iter:
        curr_h = curr_node()
        searched.append([curr_node.x, curr_node.y])
        count += 1
        min_h = curr_h
        min_pair = (0, 0)
        min_node = curr_node
        for pair in curr_node.neighbors:
            a = curr_node.x + pair[0]
            b = curr_node.y + pair[1]
            weight = curr_node.neighbors[pair]
            other_node = graph.nodes[a, b]
            other_h = other_node()
            if other_h < min_h:
                min_h = other_h
                min_pair = pair
                min_node = other_node
        if min_pair == (0, 0):
            prev_node = curr_node.prev
            prev_pair = (curr_node.x-prev_node.x, curr_node.y-prev_node.y)
            del prev_node.neighbors[prev_pair]
            curr_node = prev_node
        else:
            min_node.prev = curr_node
            curr_node = min_node

    if False and len(Q) == 0:
        print("No solution found")
        return graph, np.array([[start_node.x, start_node.y]]), np.array(searched), 0
    path = [[curr_node.x, curr_node.y]]
    path_length = 0
    while curr_node != start_node:
        path.insert(0, [curr_node.prev.x, curr_node.prev.y])
        path_length += np.linalg.norm([curr_node.x -
                                      curr_node.prev.x, curr_node.y-curr_node.prev.y])
        curr_node = curr_node.prev

    return graph, np.array(path), np.array(searched), path_length


start_pt = np.array([-3, 0])
goal_pt = np.array([3, 0])
#obstacles = [plt.Rectangle([-1,-1],2,2)]
# img, img_start_pt, img_goal_pt = convert_obstacles_to_image(
#    obstacles, (40, 40), start_pt, goal_pt)
img_cp = img.copy()
h_name = "grid dist"
graph = convert_img_to_graph(img_cp, h_grid_dist)

start = time.time()
new_graph, path, searched, path_length = a_star_2(
    graph, img_start_pt, img_goal_pt)
end = time.time()

print(h_name, end-start)
fig, ax = plt.subplots()

im = ax.imshow(img_cp)
ax.set_title("heuristic: {} \nlength: {} \nsearched: {} \ntime: {}".format(
    h_name, path_length, len(searched), end-start))
ax.plot(searched[:, 0], searched[:, 1], 'x')
ax.plot(path[:, 0], path[:, 1])
ax.plot(img_start_pt[0], img_start_pt[1], 'gs')
ax.plot(img_goal_pt[0], img_goal_pt[1], 'r*')
