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


def turn_left_till_free(pos, theta, ds, obstacles):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    # turn left until we see free space
    while any(collision(new_pos, obs) for obs in obstacles):
        time_since_collision = 0
        theta += 0.1
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        new_pos = pos + dir_vec*ds
        # ax2.plot(new_pos[0], new_pos[1], 'g.')
    return theta


def turn_right_till_wall(pos, theta, ds, obstacles):
    dir_vec = np.array([np.cos(theta), np.sin(theta)])
    new_pos = pos + dir_vec*ds
    # turn left until we see free space
    while not any(collision(new_pos, obs) for obs in obstacles):
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


def find_boundary(obstacles):
    pos = np.array([0., 0.])
    goal_pt = np.array([2*np.pi, 0])
    mode = "init"
    ds = 0.01
    path_length = 0
    theta = 0
    mode_changed = True
    obs_index = 0
    zero_crossings = []
    prev_mode = "none"
    while abs(pos[0]-goal_pt[0]) > ds:
        should_yield = True
        if mode != prev_mode:
            print("mode: {} -> {}, {}".format(prev_mode, mode, pos))
            prev_mode = mode
            mode_changed = False

        if mode == "goal":
            should_yield = False
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            if any(collision(new_pos, obs) for obs in obstacles):
                if any(abs(crossing[0]-new_pos[0]) < 2*ds for crossing in zero_crossings):
                    mode = "cross_obstacle_then_goal"
                    mode_changed = True
                else:
                    mode = "wall_start"
                    mode_changed = True
                    if all(crossing[0] > pos[0] for crossing in zero_crossings):
                        zero_crossings = [pos.copy()]
                    else:
                        zero_crossings.append(pos.copy())
                    print("zero crossing: {}".format(pos))

            else:
                pos, path_length = step_forward(
                    pos, theta, ds, path_length)

            pass
        elif mode == "wall_start":
            theta = turn_right_till_wall(pos, theta, ds, obstacles)
            theta = turn_left_till_free(pos, theta, ds, obstacles)
            wall_start_pos = pos
            close_pt = pos
            close_dist = np.linalg.norm(close_pt-goal_pt)
            pos, path_length = step_forward(
                pos, theta, ds*1.1, path_length)
            mode = "wall"
        elif mode == "wall":
            # turn left until we see free space
            theta = turn_left_till_free(pos, theta, ds, obstacles)
            theta = turn_right_till_wall(pos, theta, ds, obstacles)
            theta += 0.1
            old_pos = pos.copy()
            pos, path_length = step_forward(pos, theta, ds, path_length)
            # print((old_pos[1]+np.pi)%(2*np.pi)-np.pi, (pos[1]+np.pi) % (2*np.pi)-np.pi)
            if abs(old_pos[1] % (2*np.pi)-pos[1] % (2*np.pi)) > (2*np.pi-ds):
                zero_crossings.append(pos.copy())
                print("zero crossing: {}".format(pos))
            curr_dist = np.linalg.norm(pos-goal_pt)
            if curr_dist < close_dist:
                close_dist = curr_dist
                close_pt = pos
            if np.linalg.norm(pos-wall_start_pos) < 2*ds:
                mode = "cross_obstacle_then_goal"
            if np.linalg.norm(pos - np.array([0, 2*np.pi+ds]) - wall_start_pos) < 2*ds:
                mode = "cross_obstacle_then_wall"
            if np.linalg.norm(pos - np.array([0, -2*np.pi+ds]) - wall_start_pos) < 2*ds:
                mode = "goal"

            pass
        elif mode == "cross_obstacle_then_wall":
            theta = 0
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            pos, path_length = step_forward(pos, theta, ds, path_length)
            if not any(collision(pos, obs) for obs in obstacles):
                mode = "wall_start"
        elif mode == "cross_obstacle_then_goal":
            should_yield = False
            theta = 0
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            pos, path_length = step_forward(pos, theta, ds, path_length)

            if not any(collision(pos, obs) for obs in obstacles):
                mode = "goal"
                obs_index += 1
        elif mode == "init":
            should_yield = False
            theta = 0
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            pos, path_length = step_forward(pos, theta, ds, path_length)
            if not any(collision(pos, obs) for obs in obstacles):
                mode = "goal"
                goal_pt = np.array([new_pos[0]+2*np.pi, 0])
                mode_changed = True
        # if should_yield:
        yield pos, obs_index


# %%


boundary_t1 = []
boundary_t2 = []


def anim_func(data):
    pos, obj_index = data
    boundary_t1.append(pos[0])
    boundary_t2.append(pos[1])
    joints = fk(pos)
    l1.set_data(joints[:, 0], joints[:, 1])
    p2.set_data(pos[0], pos[1])
    l2.set_data(boundary_t1, boundary_t2)


circle = geom.Point(11, 11).buffer(4)
quad_3 = geom.Polygon(np.array(
    [[-0.000001, -0.000001], [-21, -0.000001], [-21, -21], [-0.0000001, -21]]))
circle_2 = geom.Point(5, 5).buffer(4)
circle_3 = geom.Point(5, 0).buffer(1)

obstacles = [circle, circle_3]
fig, [ax1, ax2] = plt.subplots(2, 1)
patches = []
for obs in obstacles:
    patch = dc.PolygonPatch(obs)
    patches.append(patch)
    ax1.add_patch(patch)


l1, = ax1.plot([], [], 'b')
l2, = ax2.plot([], [])
p2, = ax2.plot([], [], 'k.')

ax1.set_xlim(-10, 20)
ax1.set_ylim(-10, 16)
# ax1.axis('equal')

ax2.set_xlim(-20, 20)
ax2.set_ylim(-20, 20)

y = find_boundary(obstacles)
animation = anim.FuncAnimation(fig, anim_func, y, interval=2)
plt.show()

# %%
obstacles = [circle, quad_3]
fig, ax = plt.subplots()

y = find_boundary(obstacles)
pos, obs_index = next(y)
boundary_pts = [[pos.copy()]]
for pos, obs_index in y:
    if np.linalg.norm(pos-boundary_pts[-1]) > 0.05:
        try:
            boundary_pts[obs_index].append(pos.copy())
        except IndexError:
            boundary_pts.append([])
            boundary_pts[obs_index].append(pos.copy())

obs_q = []
boundary_pts = np.array(boundary_pts)
for boundary in boundary_pts:
    obs_q.append(geom.Polygon(boundary))

offsets = [2*np.pi, 0, -2*np.pi]
# offsets = [0]
for x_offset in offsets:
    for y_offset in offsets:
        for boundary in boundary_pts:
            boundary = np.array(boundary)
            new_bound = np.array(
                [boundary[:, 0]+x_offset, boundary[:, 1]+y_offset]).T
            polygon = geom.Polygon(new_bound)
            patch = dc.PolygonPatch(polygon)
            ax.add_patch(patch)
        # plt.plot(boundary_pts[:,0]+x_offset, boundary_pts[:,1]+y_offset)

plt.xlim([-6*np.pi, 6*np.pi])
plt.ylim([-6*np.pi, 6*np.pi])
