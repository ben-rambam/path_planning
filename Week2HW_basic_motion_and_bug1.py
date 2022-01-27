#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 13:39:32 2022

@author: jonathan
"""

import numpy as np
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(1, 1)

start_pt = np.array([-5, 4])
goal_pt = np.array([5, -4])
ax1.plot(start_pt[0], start_pt[1], 'g.')
ax1.plot(goal_pt[0], goal_pt[1], 'r.')


n = 10
obstacles = []

# circle = plt.Circle([0, 0], 0.5)
# obstacles.append(circle)
# ax1.add_patch(circle)


def make_poly(x, y, r, n):
    ts = np.sort(np.random.uniform(0, 2*np.pi, n))
    rs = np.random.uniform(r*0.9, r*1.1, n)

    pts = np.array([rs*np.cos(ts) + x, rs*np.sin(ts) + y]).T

    poly = plt.Polygon(pts)
    return poly


obs_type = "polygons"

for i in range(n):
    x = np.random.uniform(-5, 5)
    y = np.random.uniform(-5, 5)
    r = np.random.uniform(0.5, 1)
    if obs_type == "circles":
        patch = plt.Circle([x, y], r)
    elif obs_type == "polygons":
        patch = make_poly(x, y, r, 6)
    while patch.contains_point(start_pt) or patch.contains_point(goal_pt):
        print("retry making obstacle")
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        r = np.random.uniform(0.5, 1)
        if obs_type == "circles":
            patch = plt.Circle([x, y], r)
        elif obs_type == "polygons":
            patch = make_poly(x, y, r, 6)

    obstacles.append(patch)
    ax1.add_patch(patch)

ax1.set_xlim(-5, 5)
ax1.set_ylim(-5, 5)
ax1.axis('equal')


def basic_motion():
    pos = start_pt
    ds = 0.1
    path_x = [pos[0]]
    path_y = [pos[1]]
    path_length = 0

    goal_vec = goal_pt-pos
    theta = np.arctan2(goal_vec[1], goal_vec[0])
    time_since_collision = 0

    while np.linalg.norm(pos-goal_pt) > ds:
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        new_pos = pos + dir_vec*ds

        while any(obs.contains_point(ax1.transData.transform(new_pos)) for obs in obstacles):
            time_since_collision = 0
            theta += 0.1
            dir_vec = np.array([np.cos(theta), np.sin(theta)])
            new_pos = pos + dir_vec*ds
            ax1.plot(new_pos[0], new_pos[1], 'r.')

        if time_since_collision >= 10:
            goal_vec = goal_pt-pos
            theta = np.arctan2(goal_vec[1], goal_vec[0])
        time_since_collision += 1
        # ax1.plot(new_pos[0], new_pos[1], 'b.')
        pos = new_pos
        path_x.append(pos[0])
        path_y.append(pos[1])
        path_length += ds
        if path_length > 50:
            raise RuntimeError
        ax1.plot(path_x, path_y, 'b')


def turn_left_till_free(pos, theta, ds, obstacles):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    # turn left until we see free space
    while any(obs.contains_point(ax1.transData.transform(new_pos)) for obs in obstacles):
        time_since_collision = 0
        theta += 0.1
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        new_pos = pos + dir_vec*ds
        ax1.plot(new_pos[0], new_pos[1], 'g.')
    return theta


def turn_right_till_wall(pos, theta, ds, obstacles):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    # turn left until we see free space
    while not any(obs.contains_point(ax1.transData.transform(new_pos)) for obs in obstacles):
        time_since_collision = 0
        theta -= 0.1
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        new_pos = pos + dir_vec*ds
        ax1.plot(new_pos[0], new_pos[1], 'r.')
    return theta


def step_forward(pos, theta, ds, path_length):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    pos = new_pos
    path_length += ds
    if path_length > 100:
        raise RuntimeError
    return pos, path_length


def bug_1(color):
    pos = start_pt
    ds = 0.1
    path_x = [pos[0]]
    path_y = [pos[1]]
    path_length = 0

    goal_vec = goal_pt-pos
    theta = np.arctan2(goal_vec[1], goal_vec[0])
    time_since_collision = 0
    mode = "goal"

    while np.linalg.norm(pos-goal_pt) > ds:
        if mode == "goal":
            goal_vec = goal_pt-pos
            theta = np.arctan2(goal_vec[1], goal_vec[0])
            dir_vec = np.array([np.cos(theta), np.sin(theta)])
            new_pos = pos + dir_vec*ds
            if any(obs.contains_point(ax1.transData.transform(new_pos)) for obs in obstacles):
                mode = "wall_start"

            else:
                pos, path_length = step_forward(pos, theta, ds, path_length)
                path_x.append(pos[0])
                path_y.append(pos[1])

            pass
        elif mode == "wall_start":
            theta = turn_left_till_free(pos, theta, ds, obstacles)
            wall_start_pos = pos
            close_pt = pos
            close_dist = np.linalg.norm(close_pt-goal_pt)
            pos, path_length = step_forward(pos, theta, ds*1.1, path_length)
            path_x.append(pos[0])
            path_y.append(pos[1])
            mode = "wall"
        elif mode == "wall":
            # turn left until we see free space
            theta = turn_left_till_free(pos, theta, ds, obstacles)
            theta = turn_right_till_wall(pos, theta, ds, obstacles)
            theta += 0.1
            pos, path_length = step_forward(pos, theta, ds, path_length)
            path_x.append(pos[0])
            path_y.append(pos[1])
            curr_dist = np.linalg.norm(pos-goal_pt)
            if curr_dist < close_dist:
                close_dist = curr_dist
                close_pt = pos
            if np.linalg.norm(pos-wall_start_pos) < ds:
                mode = "wall_finish"

            pass
        elif mode == "wall_finish":
            theta = turn_left_till_free(pos, theta, ds, obstacles)
            theta = turn_right_till_wall(pos, theta, ds, obstacles)
            theta += 0.1
            pos, path_length = step_forward(pos, theta, ds, path_length)
            path_x.append(pos[0])
            path_y.append(pos[1])
            if np.linalg.norm(pos-close_pt) < ds:
                mode = "goal"
    ax1.plot(path_x, path_y, color)


bug_1('k')
"""
drive straight
if collision, 
    turn left until no collision
    set detour_counter to 10
if detour_counter reaches zero
    set course to path
"""
