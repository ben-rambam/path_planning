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


class Bug1():
    def __init__(self, start_pt, goal_pt, ds, obstacles):
        self.pos = start_pt
        self.goal_pt = goal_pt
        self.ds = ds
        self.obstacles = obstacles

    def update(self):
        pos = self.pos
        goal_pt = self.goal_pt
        mode = "goal"
        ds = self.ds
        path_length = 0
        theta = 0
        path_x = [pos[0]]
        path_y = [pos[1]]
        obstacles = self.obstacles
        while np.linalg.norm(pos-goal_pt) > ds:
            if mode == "goal":
                goal_vec = goal_pt-pos
                theta = np.arctan2(goal_vec[1], goal_vec[0])
                dir_vec = np.array([np.cos(theta), np.sin(theta)])
                new_pos = pos + dir_vec*ds
                if any(collision(new_pos, obs) for obs in obstacles):
                    mode = "wall_start"

                else:
                    pos, path_length = step_forward(
                        pos, theta, ds, path_length)
                    path_x.append(pos[0])
                    path_y.append(pos[1])

                pass
            elif mode == "wall_start":
                theta = turn_left_till_free(pos, theta, ds, obstacles)
                wall_start_pos = pos
                close_pt = pos
                close_dist = np.linalg.norm(close_pt-goal_pt)
                pos, path_length = step_forward(
                    pos, theta, ds*1.1, path_length)
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

            yield pos, theta, path_x, path_y


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


def find_boundary_3(obstacles):
    pos = np.array([0., 0.])
    goal_pt = np.array([2*np.pi, 0])
    mode = "goal"
    ds = 0.01
    path_length = 0
    theta = 0
    path_x = []
    path_y = []
    partway_done = False
    while np.linalg.norm(pos-goal_pt) > ds:
        should_yield = True
        if mode == "goal":
            # should_yield = False
            goal_vec = goal_pt-pos
            theta = np.arctan2(goal_vec[1], goal_vec[0])
            dir_vec = np.array([np.cos(theta), np.sin(theta)])
            new_pos = pos + dir_vec*ds
            if any(collision(new_pos, obs) for obs in obstacles):
                mode = "wall_start"

            else:
                pos, path_length = step_forward(
                    pos, theta, ds, path_length)
                # path_x.append(pos[0])
                # path_y.append(pos[1])

            pass
        elif mode == "wall_start":
            theta = turn_right_till_wall(pos, theta, ds, obstacles)
            theta = turn_left_till_free(pos, theta, ds, obstacles)
            wall_start_pos = pos
            close_pt = pos
            close_dist = np.linalg.norm(close_pt-goal_pt)
            pos, path_length = step_forward(
                pos, theta, ds*1.1, path_length)
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
                mode = "cross_obstacle"
            print(pos-wall_start_pos, np.linalg.norm(pos -
                  np.array([0, 2*np.pi]) - wall_start_pos))
            if np.linalg.norm(pos - np.array([0, 2*np.pi]) - wall_start_pos) < ds:
                partway_done = True
                mode = "cross_obstacle"
            if np.linalg.norm(pos - np.array([0, -2*np.pi]) - wall_start_pos) < ds:
                mode = "goal"

            pass
        elif mode == "cross_obstacle":
            theta = 0
            dir_vec = np.array([1, 0])
            new_pos = pos + dir_vec*ds

            pos, path_length = step_forward(pos, theta, ds, path_length)
            if partway_done:
                path_x.append(pos[0])
                path_y.append(pos[1])
            if not any(collision(pos, obs) for obs in obstacles):
                if partway_done:
                    mode = "wall_start"
                else:
                    mode = "goal"
        if should_yield:
            yield pos, theta, path_x, path_y


def find_boundary_2(obstacle):
    thetas = np.array([0., 0.])
    joints = fk(thetas)
    manipulator = geom.LineString(joints)
    step = 0.01

#     mode = "right down"
    mode = "general"
    contact_dir = np.array([1., 0.])
    noncontact_dir = np.array([0.0, -1.0])
    rotate_left_90 = np.array([[0, -1], [1, 0]])
    print("contact:{}".format(contact_dir))
    print("noncontact:{}".format(noncontact_dir))
    num_turns = 0
    first_time = True
    first_yield = thetas
    while (num_turns < 4 or np.linalg.norm(first_yield-thetas) > step) and num_turns < 5:
        if mode == "general":
            stop_early = False
            start = thetas.copy()
            if abs(contact_dir[0]) < 0.5:
                which_theta = 1
            else:
                which_theta = 0
            print("contact theta:{}".format(which_theta))
            while not manipulator.intersects(obstacle):
                #                 print("go contact:{}".format(contact_dir))

                thetas += contact_dir*step
                joints = fk(thetas)
                manipulator = geom.LineString(joints)
                if abs(start[which_theta]-thetas[which_theta]) > 2*np.pi:
                    thetas[which_theta] -= 2*np.pi*contact_dir[which_theta]
                    contact_dir = rotate_left_90@contact_dir
                    noncontact_dir = rotate_left_90@noncontact_dir
                    print("contact:{}".format(contact_dir))
                    print("noncontact:{}".format(noncontact_dir))
                    stop_early = True
                    num_turns += 1
                    break
                # yield joints, thetas
            if stop_early:
                continue
            start = thetas.copy()
            if abs(noncontact_dir[0]) < 0.5:
                which_theta = 1
            else:
                which_theta = 0
            print("noncontact theta:{}".format(which_theta))
            while manipulator.intersects(obstacle):
                #                 print("go noncontact:{}".format(noncontact_dir))
                thetas += noncontact_dir*step
                joints = fk(thetas)
                manipulator = geom.LineString(joints)
                if abs(start[which_theta]-thetas[which_theta]) > 2*np.pi:
                    # thetas[which_theta] -= 2*np.pi*noncontact_dir[which_theta]
                    contact_dir = rotate_left_90@contact_dir
                    noncontact_dir = rotate_left_90@noncontact_dir
                    print("contact:{}".format(contact_dir))
                    print("noncontact:{}".format(noncontact_dir))
                    num_turns += 1
                    break
                # yield joints,thetas
            if first_time:
                first_yield = thetas.copy()
                first_time = False
            yield joints, thetas


# %%


boundary_t1 = []
boundary_t2 = []


def anim_func_3(data):
    pos, theta, path_x, path_y = data
    joints = fk(pos)
    l1.set_data(joints[:, 0], joints[:, 1])
    p2.set_data(pos[0], pos[1])
    l2.set_data(path_x, path_y)


def anim_func(data):
    joints = data[0]
    thetas = data[1]
    l1.set_data(joints[:, 0], joints[:, 1])
    p2.set_data(thetas[0], thetas[1])
    boundary_t1.append(thetas[0])
    boundary_t2.append(thetas[1])
    l2.set_data(boundary_t1, boundary_t2)

    return l1, p2, l2


circle = geom.Point(11, 11).buffer(4)
quad_3 = geom.Polygon(np.array(
    [[-0.000001, -0.000001], [-21, -0.000001], [-21, -21], [-0.0000001, -21]]))
circle_2 = geom.Point(5, 5).buffer(4)

obstacles = [circle]
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

y = find_boundary_3(obstacles)
animation = anim.FuncAnimation(fig, anim_func_3, y, interval=2)
plt.show()

# %%
obstacles = [circle_2]
fig, ax = plt.subplots()
for obs in obstacles:
    y = find_boundary_2(obs)
    joints, thetas = next(y)
    boundary_pts = [thetas.copy()]
    for joints, thetas in y:
        if np.linalg.norm(thetas-boundary_pts[-1]) > 0.05:
            boundary_pts.append(thetas.copy())
    boundary_pts = np.array(boundary_pts)
    obs_q = geom.Polygon(boundary_pts)
    for x_offset in [2*np.pi, 0, -2*np.pi]:
        for y_offset in [2*np.pi, 0, -2*np.pi]:
            new_bound = np.array(
                [boundary_pts[:, 0]+x_offset, boundary_pts[:, 1]+y_offset]).T
            polygon = geom.Polygon(new_bound)
            patch = dc.PolygonPatch(polygon)
            ax.add_patch(patch)
            # plt.plot(boundary_pts[:,0]+x_offset, boundary_pts[:,1]+y_offset)

plt.xlim([-6*np.pi, 6*np.pi])
plt.ylim([-6*np.pi, 6*np.pi])
