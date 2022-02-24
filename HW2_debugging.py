#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:14:01 2022

@author: jonathan
"""
import descartes as dc
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import shapely.geometry as geom


a1 = 10
a2 = 10
base = (0, 0)


def fk(thetas):
    t1 = thetas[0]
    t2 = thetas[1]
    x1 = a1*np.cos(t1)
    y1 = a1*np.sin(t1)

    x2 = x1 + a2*np.cos(t1+t2)
    y2 = y1 + a2*np.sin(t1+t2)

    return np.array([(0, 0), (x1, y1), (x2, y2)])


def turn_left_till_free(pos, theta, ds, obs):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    # turn left until we see free space
    while collision(new_pos, obs):
        time_since_collision = 0
        theta += 0.1
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        new_pos = pos + dir_vec*ds
        # ax2.plot(new_pos[0], new_pos[1], 'g.')
    return theta


def turn_right_till_wall(pos, theta, ds, obs):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    # turn left until we see free space
    while not collision(new_pos, obs):
        time_since_collision = 0
        theta -= 0.1
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        new_pos = pos + dir_vec*ds
        # ax2.plot(new_pos[0], new_pos[1], 'r.')
    return theta


def step_forward(pos, theta, ds, path_length):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    pos = new_pos
    path_length += ds
    if path_length > 100:
        raise RuntimeError
    return pos, path_length


def collision(thetas, obstacle):
    joints = fk(thetas)
    manipulator = geom.LineString(joints)
    return manipulator.intersects(obstacle)


def find_boundary(obs):
    pos = np.array([0., 0.])
    goal_pt = np.array([2*np.pi, 0])
    mode = "init"
    ds = 0.01
    path_length = 0
    theta = 0
    prev_mode = "none"
    while np.linalg.norm(pos-goal_pt) > ds:
        should_yield = True
        if mode != prev_mode:
            # print("mode: {} -> {}, {}, {}".format(prev_mode, mode, pos, pos-goal_pt))
            prev_mode = mode

        if mode == "goal":
            should_yield = False
            dir_vec = np.array([1, 0])
            theta = 0
            new_pos = pos + dir_vec*ds

            if collision(new_pos, obs):
                mode = "wall_start"

            else:
                pos, path_length = step_forward(
                    pos, theta, ds, path_length)
        elif mode == "wall_start":
            theta = turn_right_till_wall(pos, theta, ds, obs)
            theta = turn_left_till_free(pos, theta, ds, obs)
            wall_start_pos = pos
            pos, path_length = step_forward(
                pos, theta, ds*1.1, path_length)
            mode = "wall"
        elif mode == "wall":
            theta = turn_left_till_free(pos, theta, ds, obs)
            theta = turn_right_till_wall(pos, theta, ds, obs)
            theta += 0.1
            old_pos = pos.copy()
            pos, path_length = step_forward(pos, theta, ds, path_length)

            curr_dist = np.linalg.norm(pos-goal_pt)

            if np.linalg.norm(pos-wall_start_pos) < ds:
                mode = "cross_obstacle_then_goal"
            if np.linalg.norm(pos - np.array([0, 2*np.pi+ds]) - wall_start_pos) < ds:
                pos = wall_start_pos + np.array([0, 2*np.pi+ds])
                mode = "cross_obstacle_then_wall"
            if np.linalg.norm(pos + np.array([0, 2*np.pi+ds]) - wall_start_pos) < ds:
                pos = wall_start_pos - np.array([0, 2*np.pi+ds])
                mode = "goal"
        elif mode == "cross_obstacle_then_wall":
            theta = 0
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            pos, path_length = step_forward(pos, theta, ds, path_length)
            if not collision(pos, obs):
                mode = "wall_start"
        elif mode == "cross_obstacle_then_goal":
            should_yield = False
            theta = 0
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            pos, path_length = step_forward(pos, theta, ds, path_length)

            if not collision(pos, obs):
                mode = "goal"
        elif mode == "init":
            should_yield = False
            theta = 0
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            pos, path_length = step_forward(pos, theta, ds, path_length)
            if not collision(pos, obs):
                mode = "goal"
                goal_pt = np.array([new_pos[0]+2*np.pi, 0])
        if should_yield:
            yield pos


# %%
circle = geom.Point(11, 11).buffer(4)
quad_3 = geom.Polygon(np.array(
    [[-0.000001, -0.000001], [-21, -0.000001], [-21, -21], [-0.0000001, -21]]))
circle_2 = geom.Point(5, 5).buffer(4)
circle_3 = geom.Point(5, 0).buffer(1)
# %%


boundary_t1 = []
boundary_t2 = []


def anim_func(data):
    pos = data
    boundary_t1.append(pos[0])
    boundary_t2.append(pos[1])
    joints = fk(pos)
    l1.set_data(joints[:, 0], joints[:, 1])
    p2.set_data(pos[0], pos[1])
    l2.set_data(boundary_t1, boundary_t2)


obs = circle
fig, [ax1, ax2] = plt.subplots(2, 1)
patch = dc.PolygonPatch(obs)
ax1.add_patch(patch)

l1, = ax1.plot([], [], 'b')
l2, = ax2.plot([], [])
p2, = ax2.plot([], [], 'k.')

ax1.set_xlim(-10, 20)
ax1.set_ylim(-10, 16)
# ax1.axis('equal')

ax2.set_xlim(-20, 20)
ax2.set_ylim(-20, 20)

y = find_boundary(circle)
animation = anim.FuncAnimation(fig, anim_func, y, interval=2)
plt.show()

# %%
obstacles = [circle_3, circle, circle_2, quad_3]
fig, ax = plt.subplots()


boundary_pts = []
for obs in obstacles:
    y = find_boundary(obs)
    boundary = [pos for pos in y]
    boundary_pts.append(np.array(boundary))

offsets = [2*np.pi, 0, -2*np.pi]
offsets = [0]
for x_offset in offsets:
    for y_offset in offsets:
        for boundary in boundary_pts:
            boundary = np.array(boundary)
            new_bound = np.array(
                [boundary[:, 0]+x_offset, boundary[:, 1]+y_offset]).T
            plt.plot(new_bound[:, 0], new_bound[:, 1])
            # polygon = geom.Polygon(new_bound)
            # patch = dc.PolygonPatch(polygon)
            # ax.add_patch(patch)
        # plt.plot(boundary_pts[:,0]+x_offset, boundary_pts[:,1]+y_offset)

plt.xlim([-6*np.pi, 6*np.pi])
plt.ylim([-6*np.pi, 6*np.pi])
