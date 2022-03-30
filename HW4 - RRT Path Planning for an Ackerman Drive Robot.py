#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 09:55:39 2022

@author: jonathan
"""

import copy
import heapq
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import shapely.geometry as geom
import descartes as dc

start_q = (5, 5, np.pi/4)
delta_t = 5


def fk(t, state, u):
    x = state[0]
    y = state[1]
    theta = state[2]

    s = 1
    k = u[0]

    xdot = s*np.cos(theta)
    ydot = s*np.sin(theta)
    thetadot = k*s

    return [xdot, ydot, thetadot]


max_k = 0.5
min_k = -max_k

ks = np.linspace(min_k, max_k, 11)
ts = np.linspace(0, delta_t)

trajectories = []
for k in ks:
    res = scipy.integrate.solve_ivp(
        fk, [0, delta_t], start_q, t_eval=ts, args=[[k]])
    trajectories.append(res.y)

plt.figure()
for traj in trajectories:
    plt.plot(traj[0], traj[1])
plt.axis('equal')

# %%
obs_pts = [[8, 6], [10, 7], [10, 10], [6, 10], [6, 9]]
obs = geom.Polygon(obs_pts)
obs_patch = dc.PolygonPatch(obs)


def collision(t, state, u):
    pt = geom.Point(state[0], state[1])
    if obs.contains(pt):
        return -1.0
    else:
        return 1.0


collision.terminal = True
# collision.direction = -1

trajectories = []
for k in ks:
    res = scipy.integrate.solve_ivp(fk, [0, delta_t], start_q, max_step=0.2, args=[
                                    [k]], events=[collision])
    trajectories.append(res.y)

fig, ax = plt.subplots()
ax.add_patch(obs_patch)
for traj in trajectories:
    ax.plot(traj[0], traj[1])

# %%


def arc_dist(q1, q2):
    p1x = q1[0]
    p2x = q2[0]
    p1y = q1[1]
    p2y = q2[1]

    u1x = np.cos(q1[2])
    u1y = np.sin(q1[2])
    u2x = np.cos(q2[2])
    u2y = np.sin(q2[2])

    delpx = p1x-p2x
    delpy = p1y-p2y

    b = 2*delpx*u2y-2*delpy*u2x
    c = delpx**2 + delpy**2

    k = -b/c

    cx = p2x - u2y/k
    cy = p2y + u2x/k
    numerator = (p1x-cx)*(p2y-cy) - (p1y-cy)*(p2x-cx)
    denominator = (p1x-cx)*(p2x-cx) + (p1y-cy)*(p2y-cy)
    # temp = numerator*k**2
    # phi = np.arcsin(min(max(temp, -1), 1))
    phi = np.arctan2(numerator, denominator)

    d = np.remainder(phi/k, np.abs(2*np.pi/k))

    theta1 = q1[2]
    theta2 = q2[2]
    tangle = theta2-phi

    omega = theta1-tangle

    if abs(k) > max_k:
        d += 100*(abs(k)-max_k)/max_k

    return d + 3*abs(omega)


print(arc_dist((0, 0, np.pi/2), (10, 0, np.pi/2)))
print(arc_dist((0, 0, np.pi/2), (5, 5, 0)))
print(arc_dist((0, 0, np.pi/2-0.1), (-5, 5, np.pi)))
print(arc_dist((0, 0, np.pi/2), (-5, -5, 0)))


# %%


class Node():
    def __init__(self, xy, hfunc):
        self.neighbors = {}
        self.g = np.Inf
        self.xy = np.array(xy)
        self.hfunc = hfunc
        self.terminal = False

    def __lt__(self, other):
        lt = False
        f1 = self.g+self.hfunc(self.xy)
        f2 = other.g+self.hfunc(self.xy)
        if np.abs(f1-f2) < 0.001:
            #print("almost equal")
            lt = self.g > other.g
        else:
            lt = f1 < f2

        #lt = f1 < f2
        return lt

    def __eq__(self, other):
        return all(self.xy == other.xy)

    def __call__(self):
        return self.hfunc(np.array([self.x, self.y]))
        try:
            return self.f
        except:
            self.f = self.g+self.hfunc(np.array([self.x, self.y]))
            return self.f


class Graph():
    def __init__(self, dist_func):
        self.nodes = {}
        self.dist_func = dist_func

    def __iadd__(self, t):
        n1 = t[0]
        n2 = t[1]
        u = t[2]
        self.connect(n1, n2, u)
        return self

    def connect(self, n1, n2, u):
        try:
            self.nodes[n1].neighbors[n2] = u
        except KeyError:
            self.nodes[n1] = Node(n1, self.hfunc)
            self.nodes[n1].neighbors[n2] = u
        try:
            self.nodes[n2].prev = n1
        except KeyError:
            self.nodes[n2] = Node(n2, self.hfunc)
            self.nodes[n2].prev = n1

    def draw(self, ax):
        for pt in self.nodes:
            node = self.nodes[pt]
            for neighbor in node.neighbors:
                ax.plot([pt[0], neighbor[0]], [pt[1], neighbor[1]], 'k')

    def hfunc(self, node):
        return self.dist_func(node, self.goal)

    def nearest(self, q):
        min_node = None
        min_dist = np.Inf
        for n in self.nodes:
            if self.nodes[n].terminal:
                continue
            dist = self.dist_func(n, q)
            if dist < min_dist:
                min_node = n
                min_dist = dist
        return min_node


def SE2dist(q1, q2):
    pos_dist = np.sqrt((q1[0]-q2[0])**2 + (q1[1]-q2[1])**2)
    angle_dist = np.abs(q1[2]-q2[2])

    return np.linalg.norm(np.array(q1)-np.array(q2))


g = Graph(arc_dist)
for i in range(len(ks)):
    print(i, trajectories[i][:, -1])
    g += (start_q, tuple(trajectories[i][:, -1]), ks[i])

target_pt = (10, 6, np.pi)
near_pt = g.nearest(target_pt)

obs_patch = dc.PolygonPatch(obs)
fig, ax = plt.subplots()
ax.add_patch(obs_patch)
for traj in trajectories:
    ax.plot(traj[0], traj[1])
ax.scatter(10, 6, label="target")
ax.scatter(near_pt[0], near_pt[1], label="nearest")
plt.legend()

# %%


def step(q1, k, timestep):
    res = scipy.integrate.solve_ivp(fk, [0, timestep], q1, max_step=0.2, args=[
                                    [k]], events=[collision])
#     print(res)
    collided = len(res.t_events[0]) > 0
    return res.y, collided


def get_best_control(q1, q2, timestep):
    collided = False

    def final_dist(k):
        #         print(k)
        s, collided = step(q1, k, timestep)
#         print(s)

        final_q = s[:, -1]
        return arc_dist(final_q, q2)

    res = scipy.optimize.minimize(
        final_dist, 0, bounds=[(min_k, max_k)], tol=0.001, options={"maxiter": 100})
    best_traj, collided = step(q1, res.x, timestep)
    return res.x, best_traj, collided


# ks = np.linspace(min_k, max_k, 11)
#
# fig,ax = plt.subplots()
# for k in ks:
#     q1 = near_pt
#     q2 = target_pt
#     timestep = 1
#     def final_dist(k):
#         res = scipy.integrate.solve_ivp(fk, [0,timestep], q1, max_step=0.2, args=[[k]], events=[collision])
#         ax.plot(res.y[0], res.y[1])
#         final_q = res.y[:,-1]
#         ax.scatter(res.y[0,-1],res.y[1,-1])
#         return arc_dist(final_q,q2)
#     print(final_dist(k))


best_k, best_traj, collided = get_best_control(near_pt, target_pt, 1)

obs_patch = dc.PolygonPatch(obs)
fig, ax = plt.subplots()
ax.add_patch(obs_patch)
for traj in trajectories:
    ax.plot(traj[0], traj[1])
ax.scatter(10, 6, label="target")
ax.scatter(near_pt[0], near_pt[1], label="nearest")
ax.plot(best_traj[0], best_traj[1], label="best trajectory")
plt.legend()

# %%
q1 = start_q

target_pt = (10, 6, np.pi/2)
dist = arc_dist(q1, target_pt)

fig, ax = plt.subplots()
ax.scatter(start_q[0], start_q[1])
ax.scatter(target_pt[0], target_pt[1])
trajectories = []
count = 0
while dist > 0.2 and count < 100:
    timestep = 1
    best_k, best_traj, collided = get_best_control(q1, target_pt, timestep)
#     print(best_k)
    ax.plot(best_traj[0], best_traj[1])
    trajectories.append(best_traj)
    q1 = best_traj[:, -1]
    dist = arc_dist(q1, target_pt)
    count += 1


ax.axis("equal")

# %%


def try_add(graph, target, timestep):
    near = graph.nearest(target)
    best_k, best_traj, collided = get_best_control(near, target, timestep)

    if not collided:
        new_pt = tuple(best_traj[:, -1])

        graph += (near, new_pt, best_k)

        return graph, best_traj
    else:
        return graph, None


goal_q = (12, 12, np.pi/2)
goal_q = (12, 12, np.pi)


def generate_graph():
    g = Graph(arc_dist)

    g.nodes[start_q] = Node(start_q, g.hfunc)

    near = g.nearest(goal_q)
    goal_dist = arc_dist(near, goal_q)
    min_goal_dist = np.Inf

    while goal_dist > 0.3:
        q_rand = tuple(np.random.uniform([0, 0, 0], [30, 30, 2*np.pi]))
#         print(q_rand)
        g, best_traj = try_add(g, q_rand, 1)
        if best_traj is not None:
            yield best_traj, q_rand
        g, best_traj = try_add(g, goal_q, 0.1)
        if best_traj is not None:
            yield best_traj, q_rand
        near = g.nearest(goal_q)
        goal_dist = arc_dist(near, goal_q)
        if goal_dist < min_goal_dist:
            min_goal_dist = goal_dist
            print(goal_dist)
    return


fig, ax = plt.subplots()

obs_patch = dc.PolygonPatch(obs)
ax.add_patch(obs_patch)
# rand_pt, = ax.plot(0,0,'rs')
ax.plot(start_q[0], start_q[1], 'gs')
ax.plot(goal_q[0], goal_q[1], 'rs')
ax.quiver(goal_q[0], goal_q[1], np.cos(goal_q[2]), np.sin(goal_q[2]))

ax.axis('equal')


def anim_func(data):
    traj = data[0]
#     rand_pt = data[1]
    l, = ax.plot(traj[0], traj[1])
#     rand_pt.set_data([rand_pt[0]],[rand_pt[1]])
    return l


animation = anim.FuncAnimation(
    fig, anim_func, generate_graph, interval=20, repeat=False)

# y = generate_graph()

# for val in y:
#     print(val)
