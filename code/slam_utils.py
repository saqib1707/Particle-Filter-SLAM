import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import load_data
# import pr2_utils as utils


UNITS = "m"
LOG4 = np.log(4)

LIDAR_SE3POSE_RF = np.eye(4)
LIDAR_SE3POSE_RF[0:3, 3] = np.array([330.2/2 - (330.2 - 298.33), 0, 514.35]) / 1000.0   # origin of lidar frame in robot frame
    
# LIDAR_SE3POSE_RF = array( [[1.     , 0.     , 0.     , 0.13323],
#                            [0.     , 1.     , 0.     , 0.     ],
#                            [0.     , 0.     , 1.     , 0.51435],
#                            [0.     , 0.     , 0.     , 0.     ]])

MAP_RES = 0.05
MAP_XMIN = -30.0
MAP_YMIN = -30.0
MAP_XMAX = 30.0
MAP_YMAX = 30.0

ROBOT_LENGTH = 330.20/1000 + 0.254   # (in meters)
ROBOT_WIDTH = 476.25/1000          # (in meters)
ROBOT_AVG_DIM = (ROBOT_LENGTH + ROBOT_WIDTH)/2
ROBOT_DIM_CELLS = np.ceil((ROBOT_AVG_DIM - MAP_XMIN) / MAP_RES).astype(np.int16) - 1


def match_closest_stamps(ref_stamps, nonref_stamps):
    num_ref_stamps = ref_stamps.shape[0]
    num_nonref_stamps = nonref_stamps.shape[0]
    
    match_idx = np.zeros_like(ref_stamps, dtype=np.int16)
    
    for t in range(num_ref_stamps):
        # find the index of minimum absolute time difference in nonref stamps data for each ref timestamp
        time_diff = ref_stamps[t] - nonref_stamps
        match_idx[t] = np.argmin(np.abs(time_diff))
        
    return match_idx


def match_closest_past_stamps(ref_stamps, nonref_stamps, lidar_stamps):
    num_ref_stamps = ref_stamps.shape[0]

    match_idx = np.zeros_like(ref_stamps, dtype=np.int16)

    for t in range(num_ref_stamps):
        # find the index of minimum non-negative (>=0) time difference in nonref data for each ref timestamp
        time_diff = ref_stamps[t] - nonref_stamps
        match_idx[t] = np.argmin(np.where(time_diff >= 0, time_diff, np.inf))
    
    return match_idx


def get_valid_lidar_scans(lidar_ranges):
    # remove lidar scans which are too close or too far, that is, outside the range (0.1, 30)
    valid_locs = np.where(np.logical_and(lidar_ranges >= 0.1, lidar_ranges <= 30))   # (5346827,) (5346827,)

    valid_lidar_locs = np.zeros_like(lidar_ranges)
    valid_lidar_locs[valid_locs[0], valid_locs[1]] = 1
    
    return valid_lidar_locs


def get_se3pose_from_xytheta(xytheta):
    """
        xytheta: (..., 3)
    """
    xy_vec = xytheta[..., 0:2]
    theta_vec = xytheta[..., 2]
    cos_theta_vec = np.cos(theta_vec)
    sin_theta_vec = np.sin(theta_vec)
    
    if len(xytheta.shape) == 3:
        se3_pose = np.zeros((xytheta.shape[0], xytheta.shape[1], 4, 4))
    elif len(xytheta.shape) == 2:
        se3_pose = np.zeros((xytheta.shape[0], 4, 4))
        
    se3_pose[..., :, :] = np.eye(4)
    se3_pose[..., 0, 0] = cos_theta_vec
    se3_pose[..., 1, 1] = cos_theta_vec
    se3_pose[..., 0, 1] = -sin_theta_vec
    se3_pose[..., 1, 0] = sin_theta_vec
    se3_pose[..., 0:2, 3] = xy_vec

    return se3_pose


def compute_lidar_xyz_LF(lidar_ranges, lidar_angle_min, lidar_angle_max, lidar_angle_increment):
    theta_range = np.arange(lidar_angle_min*180/np.pi, lidar_angle_max*180/np.pi + lidar_angle_increment*180/np.pi, 
                            lidar_angle_increment*180/np.pi) * np.pi / 180

    cos_theta_range = np.cos(theta_range)
    sin_theta_range = np.sin(theta_range)

    # lidar scans converted to cartesian coordinates (x,y,z) in lidar frame
    lidar_xyz_LF = np.zeros((lidar_ranges.shape[0], lidar_ranges.shape[1], 3))
    
    lidar_xyz_LF[:,:,0] = np.multiply(lidar_ranges, cos_theta_range)  # x-coord in lidar frame (in m)
    lidar_xyz_LF[:,:,1] = np.multiply(lidar_ranges, sin_theta_range)  # y-coord in lidar frame (in m)

    return lidar_xyz_LF


def compute_lidar_xyz_RF(lidar_xyz_LF):
    lidar_rotmat_RF = LIDAR_SE3POSE_RF[0:3, 0:3]
    lidar_origin_RF = LIDAR_SE3POSE_RF[0:3, 3]
    assert(lidar_rotmat_RF.shape[0] == 3 and lidar_rotmat_RF.shape[1] == 3)
    assert(lidar_origin_RF.shape[0] == 3)
    
    lidar_xyz_RF = np.matmul(lidar_xyz_LF, lidar_rotmat_RF.T) + lidar_origin_RF
    return lidar_xyz_RF
    
    
def compute_lidar_xyz_WF(lidar_xyz_RF, robot_pose_WF):
    robot_rotmat_WF = robot_pose_WF[0:3, 0:3]
    robot_origin_WF = robot_pose_WF[0:3, 3]
    assert(robot_rotmat_WF.shape[0] == 3 and robot_rotmat_WF.shape[1] == 3)
    assert(robot_origin_WF.shape[0] == 3)
    
    lidar_xyz_WF = np.matmul(lidar_xyz_RF, robot_rotmat_WF.T) + robot_origin_WF
    return lidar_xyz_WF


def compute_lidar_xyz_WF_vect(lidar_xyz_RF, robot_se3pose_WF):
    """
        lidar_xyz_RF: (4956, 1081, 3)
        robot_se3pose_WF: (4956, 4, 4)
    """
    robot_rotmat_WF = robot_se3pose_WF[..., 0:3, 0:3]
    robot_origin_WF = robot_se3pose_WF[..., 0:3, 3]
    robot_origin_WF = np.expand_dims(robot_origin_WF, axis=1)
    lidar_xyz_WF = np.matmul(lidar_xyz_RF, np.transpose(robot_rotmat_WF, (0, 2, 1))) + robot_origin_WF
    
    return lidar_xyz_WF


def get_empty_map(res=MAP_RES, xmin=MAP_XMIN, ymin=MAP_YMIN, xmax=MAP_XMAX, ymax=MAP_YMAX):
    # init MAP
    MAP = {}
    MAP['res']   = res      # meters
    MAP['xmin']  = xmin     # meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) # number of cells in horizontal
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) # number of cells in vertical
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)         # DATA TYPE: char or int8
    MAP['logodds'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)  # DATA TYPE: float
    
    return MAP

def get_texture_map(res=MAP_RES, xmin=MAP_XMIN, ymin=MAP_YMIN, xmax=MAP_XMAX, ymax=MAP_YMAX):
    # init MAP
    MAP = {}
    MAP['res']   = res      # meters
    MAP['xmin']  = xmin     # meters
    MAP['ymin']  = ymin
    MAP['xmax']  = xmax
    MAP['ymax']  = ymax
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) # number of cells in horizontal
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) # number of cells in vertical
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.float32)         # DATA TYPE: char or int8
    
    return MAP


def plot_grid_map(mp, mp_logodds, savepath):
    plt.figure(figsize=(8,8))

#     plt.subplot(1,2,1)
    plt.imshow(mp, cmap='gray', vmin=np.min(mp), vmax=np.max(mp))
    plt.ylabel("X (world-frame)", fontsize=14)
    plt.xlabel("Y (world-frame)", fontsize=14)
    plt.title("Occupancy Grid Map", fontsize=14)

#     plt.subplot(1,2,2)
#     plt.imshow(mp_logodds, cmap='gray', vmin=np.min(mp_logodds), vmax=np.max(mp_logodds))
#     plt.ylabel("X (world-frame)", fontsize=14)
#     plt.xlabel("Y (world-frame)", fontsize=14)
#     plt.title("Occupancy Grid Map (log-odds)", fontsize=14)

    plt.savefig(savepath)
    plt.show()
    plt.close()
    
    
def plot_grid_map_particles_traj(mp, traj=None, p_states=None, savepath=None):
    # plot the particle dead-reck trajectory inside the map
    mp_traj = np.zeros((mp.shape[0], mp.shape[1], 3))
    mp_traj[:,:,0] = np.copy(mp)
    mp_traj[:,:,1] = np.copy(mp)
    mp_traj[:,:,2] = np.copy(mp)

    if traj is not None:
        # convert particle traj from meters to cells coordinates
        traj_cells = np.ceil((traj[..., 0:2] - MAP_XMIN) / MAP_RES).astype(np.int16) - 1  # (4956, 2)
        mp_traj[traj_cells[..., 0], traj_cells[..., 1]] = [1,0,0]
        mp_traj[traj_cells[..., 0]-1, traj_cells[..., 1]-1] = [1,0,0]
        mp_traj[traj_cells[..., 0]+1, traj_cells[..., 1]+1] = [1,0,0]
        mp_traj[traj_cells[..., 0]-1, traj_cells[..., 1]] = [1,0,0]
        mp_traj[traj_cells[..., 0]+1, traj_cells[..., 1]] = [1,0,0]
        mp_traj[traj_cells[..., 0], traj_cells[..., 1]-1] = [1,0,0]
        mp_traj[traj_cells[..., 0], traj_cells[..., 1]+1] = [1,0,0]
        mp_traj[traj_cells[..., 0]-1, traj_cells[..., 1]+1] = [1,0,0]
        mp_traj[traj_cells[..., 0]+1, traj_cells[..., 1]-1] = [1,0,0]
    
    if p_states is not None:
        # convert particle states from meters to cells coordinates
        part_cells = np.ceil((p_states[..., 0:2] - MAP_XMIN) / MAP_RES).astype(np.int16) - 1  # (num_particles, 2)
        mp_traj[part_cells[..., 0], part_cells[..., 1]] = [0,0,1]
        mp_traj[part_cells[..., 0]-1, part_cells[..., 1]-1] = [0,0,1]
        mp_traj[part_cells[..., 0]+1, part_cells[..., 1]+1] = [0,0,1]
        mp_traj[part_cells[..., 0]-1, part_cells[..., 1]] = [0,0,1]
        mp_traj[part_cells[..., 0]+1, part_cells[..., 1]] = [0,0,1]
        mp_traj[part_cells[..., 0], part_cells[..., 1]-1] = [0,0,1]
        mp_traj[part_cells[..., 0], part_cells[..., 1]+1] = [0,0,1]
        mp_traj[part_cells[..., 0]-1, part_cells[..., 1]+1] = [0,0,1]
        mp_traj[part_cells[..., 0]+1, part_cells[..., 1]-1] = [0,0,1]

    plt.figure(figsize=(8,8))
    plt.imshow(mp_traj)
    plt.ylabel("X (world-frame)", fontsize=14)
    plt.xlabel("Y (world-frame)", fontsize=14)
    plt.title("Occupancy grid map with trajectory", fontsize=14)

    plt.savefig(savepath)
    plt.show()
    plt.close()
    
    
def plot_particle_traj(traj, savepath, stdv=0, stdw=0):
    if len(traj.shape) == 2:
        traj = np.expand_dims(traj, axis=0)

    num_particles = traj.shape[0]

    plt.figure(figsize=(16,6))
    for i in range(num_particles):
        plt.subplot(1,2,1)
        plt.plot(traj[i, :, 0], traj[i, :, 1])
        plt.xlabel("X [m]", fontsize=14)
        plt.ylabel("Y [m]", fontsize=14)
        plt.title("Particle trajectory in X-Y plane (N="+str(num_particles)+", $\sigma_v$ = "+str(stdv)+")", fontsize=14)
        plt.grid(linestyle='--')

        plt.subplot(1,2,2)
        plt.plot(traj[i, :, 2])
        plt.xlabel("time index [n]", fontsize=14)
        plt.ylabel("theta [rad]", fontsize=14)
        plt.title("Particle orientation (N="+str(num_particles)+", $\sigma_{\omega}$ = "+str(stdw)+")", fontsize=14)
        plt.grid(linestyle='--')

    plt.savefig(savepath)
    plt.show()
    plt.close()
    
    
    
def bresenham2D(sx, sy, ex, ey):
    '''
    Bresenham's ray tracing algorithm in 2D.
    Inputs:
      (sx, sy)	start point of ray
      (ex, ey)	end point of ray
    '''
    sx = int(round(sx))
    sy = int(round(sy))
    ex = int(round(ex))
    ey = int(round(ey))
    dx = abs(ex-sx)
    dy = abs(ey-sy)
    steep = abs(dy)>abs(dx)
    if steep:
        dx,dy = dy,dx # swap 

    if dy == 0:
        q = np.zeros((dx+1,1))
    else:
        q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
    if steep:
        if sy <= ey:
            y = np.arange(sy,ey+1)
        else:
            y = np.arange(sy,ey-1,-1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx,ex+1)
        else:
            x = np.arange(sx,ex-1,-1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)
    
    return np.vstack((x,y))


def gen_map_init_scan(lidar_xyz_WF, valid_lidar_scan):
    """
        lidar_xyz_WF: (1081, 3)
        LIDAR_SE3POSE_RF: (4,4) pose matrix
        valid_lidar_scan: (1081,)
    """
    MAP = get_empty_map()
    
    # define the starting point of rays and convert from meters to cells
    sx = 0 + LIDAR_SE3POSE_RF[0, 3]
    sy = 0 + LIDAR_SE3POSE_RF[1, 3]
    
    sx_cells = np.ceil((sx - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1
    sy_cells = np.ceil((sy - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
    
    # convert from meters to cells
    ex_cells = np.ceil((lidar_xyz_WF[..., 0] - MAP['xmin']) / MAP['res'] ).astype(np.int16) - 1
    ey_cells = np.ceil((lidar_xyz_WF[..., 1] - MAP['ymin']) / MAP['res'] ).astype(np.int16) - 1
    
    num_scan_ang = lidar_xyz_WF.shape[0]

    for scan_idx in tqdm(range(num_scan_ang)):
        if valid_lidar_scan[scan_idx] == False:
            continue
        
        ray_x, ray_y = bresenham2D(sx=sx_cells, sy=sy_cells, ex=ex_cells[scan_idx], ey=ey_cells[scan_idx]).astype(np.int16)

        valid_xy = np.logical_and(np.logical_and(ray_x >= 0, ray_x < MAP['sizex']), 
                                  np.logical_and(ray_y >= 0, ray_y < MAP['sizey']))

        ray_x = ray_x[valid_xy]
        ray_y = ray_y[valid_xy]

        if ray_x.shape[0] == 0:
            continue

        MAP['logodds'][ray_x[-1], ray_y[-1]] += LOG4
        MAP['logodds'][ray_x[:-1], ray_y[:-1]] -= LOG4

    # constrain log-odds map to avoid overconfident estimation
    MAP['logodds'] = np.clip(MAP['logodds'], a_min=-100, a_max=100)
    temp = 1. / (1 + np.exp(MAP['logodds']))
    MAP['map'][temp > 0.5] = 1.
    MAP['map'][temp < 0.5] = 0
    MAP['map'][temp == 0.5] = 0.5

    return MAP['map'], MAP['logodds']


def compute_robot_vel(encoder_counts, encoder_stamps, imu_ang_vel):
    # compute linear velocity using encoder data
    encoder_time_diff = encoder_stamps[1:] - encoder_stamps[0:-1]

    whl_diam = 0.254                     # (in m)
    whl_circ = np.pi * whl_diam
    dist_trav_per_tick = whl_circ / 360   # distance traveled by wheel per encoder tick measured in meter

    right_whl_dist_trav = 0.5 * (encoder_counts[:,0] + encoder_counts[:,2]) * dist_trav_per_tick
    left_whl_dist_trav = 0.5 * (encoder_counts[:,1] + encoder_counts[:,3]) * dist_trav_per_tick

    right_whl_vel = right_whl_dist_trav[1:] / encoder_time_diff
    left_whl_vel = left_whl_dist_trav[1:] / encoder_time_diff

    robot_lin_vel = np.zeros((encoder_stamps.shape[0]))
    robot_lin_vel[1:] = (right_whl_vel + left_whl_vel) / 2        # v = (vl + vr) / 2
    print("Robot linear velocity shape:", robot_lin_vel.shape)

    # compute robot angular velocity from the yaw rate provided by IMU data
    robot_ang_vel = imu_ang_vel[:,2]
    print("Robot angular velocity shape:", robot_ang_vel.shape)

    return robot_lin_vel, robot_ang_vel


def motion_model(state, robot_vel, tau_t, std_lin_vel=0, std_ang_vel=0):
    """
        Function defining robot differential drive motion model
        
        state: (num_particles, 3)
        robot_vel: (2,)
        tau_t: scalar
    """
    num_particles = state.shape[0]
    noise_mat = np.array([np.random.normal(0, std_lin_vel, (num_particles)), 
                          np.random.normal(0, std_ang_vel, (num_particles))]).T   # (num_particles, 2)
    
    robot_vel_noisy = robot_vel + noise_mat    # (num_particles, 2)

    tmp1 = robot_vel_noisy[:, 1] * tau_t / 2   # (num_particles, 1)
    tmp2 = tmp1 + state[:, 2]                  # (num_particles, 1)
    tmp3 = np.divide(np.sin(tmp1), tmp1)       # (num_particles, 1)
    
    tmp6 = np.multiply(robot_vel_noisy[:, 0], tmp3)   # (num_particles, 1)
    tmp7 = np.multiply(tmp6, np.cos(tmp2))            # (num_particles, 1)
    tmp8 = np.multiply(tmp6, np.sin(tmp2))            # (num_particles, 1)
    
    tmp9 = np.array([tmp7, tmp8, robot_vel_noisy[:, 1]]).T  # (num_particles, 3)
    next_state = state + tau_t * tmp9             # (num_particles, 3)

    return next_state


def pred_particle_traj(robot_vel, encoder_stamps, num_particles=1, std_lin_vel=0, std_ang_vel=0, tosave=False, savepath=None):
    """
        RETURNS: (num_particles, 4956, 3)
    """
    particle_state = np.zeros((num_particles, 3))
    particle_weight = np.full((num_particles), 1./num_particles)

    num_traj_steps = encoder_stamps.shape[0] - 1
    particle_traj = np.zeros((num_particles, num_traj_steps+1, 3))
    particle_traj[:, 0, :] = particle_state

    tau_t = encoder_stamps[1:] - encoder_stamps[0:-1]     # (4955,)

    for t in tqdm(range(1, num_traj_steps)):
        particle_state = motion_model(particle_state, robot_vel[t], tau_t[t], std_lin_vel=std_lin_vel, 
                                      std_ang_vel=std_ang_vel)
        particle_traj[:, t+1, :] = particle_state
    
    if tosave == True:
        plot_particle_traj(particle_traj, savepath, std_lin_vel, std_ang_vel)
   
    return particle_traj


def gen_map_dead_reck(lidar_xyz_RF, robot_traj_WF, valid_lidar_scan):
    """
        lidar_xyz_RF: (4956, 1081, 3)
        LIDAR_SE3POSE_RF: (4,4) pose matrix
        robot_traj_WF: (4956, 3)  (X, Y, theta)
    """
    
    # initialize an empty map
    MAP = get_empty_map()

    robot_se3pose_WF = get_se3pose_from_xytheta(robot_traj_WF)  # (4956, 4, 4)
    lidar_xyz_WF = compute_lidar_xyz_WF_vect(lidar_xyz_RF, robot_se3pose_WF)   # (4956, 1081, 3)
    
    # compute the starting locations of rays in WF
    sx = robot_traj_WF[:, 0] + LIDAR_SE3POSE_RF[0, 3]
    sy = robot_traj_WF[:, 1] + LIDAR_SE3POSE_RF[1, 3]
    
    # convert from meters to grid cell coordinates
    sx_cell = np.ceil((sx - MAP['xmin']) / MAP['res']).astype(np.int16) - 1   # (4956,)
    sy_cell = np.ceil((sy - MAP['ymin']) / MAP['res']).astype(np.int16) - 1   # (4956,)
    
    # convert from meters to grid cell coordinates
    ex_cell = np.ceil((lidar_xyz_WF[..., 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1  # (4956, 1081)
    ey_cell = np.ceil((lidar_xyz_WF[..., 1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1  # (4956, 1081)

    num_scan_ang = lidar_xyz_RF.shape[1]
    num_traj_steps = robot_traj_WF.shape[0]

    for traj_idx in tqdm(range(num_traj_steps)):
        for scan_idx in range(num_scan_ang):
            if valid_lidar_scan[traj_idx, scan_idx] == False:
                continue
            
            ray_x, ray_y = bresenham2D(sx=sx_cell[traj_idx], 
                                       sy=sy_cell[traj_idx], 
                                       ex=ex_cell[traj_idx, scan_idx], 
                                       ey=ey_cell[traj_idx, scan_idx]
                                      ).astype(np.int16)

            valid_xy = np.logical_and(np.logical_and(ray_x >= 0, ray_x < MAP['sizex']), 
                                      np.logical_and(ray_y >= 0, ray_y < MAP['sizey']))
            
            ray_x = ray_x[valid_xy]
            ray_y = ray_y[valid_xy]
            
            if ray_x.shape[0] == 0:
                continue

            # update the map logodds
            MAP['logodds'][ray_x[:-1], ray_y[:-1]] -= LOG4
            MAP['logodds'][ray_x[-1], ray_y[-1]] += LOG4

    # constrain log-odds map to avoid overconfident estimation
    MAP['logodds'] = np.clip(MAP['logodds'], a_min=-100, a_max=50)
    temp = 1. / (1 + np.exp(MAP['logodds']))
    MAP['map'][temp > 0.5] = 1.
    MAP['map'][temp < 0.5] = 0
    MAP['map'][temp == 0.5] = 0.5

    return MAP['map'], MAP['logodds']


def compute_lidar_xyz_WF_vect2(lidar_xyz_RF, particle_se3pose_WF):
    """
        lidar_xyz_RF: (1081, 3)
        particle_se3pose_WF: (num_particles, 4, 4)
    """
    particle_rotmat_WF = particle_se3pose_WF[..., 0:3, 0:3]
    particle_origin_WF = particle_se3pose_WF[..., 0:3, 3]
    particle_origin_WF = np.expand_dims(particle_origin_WF, axis=1)
    
    lidar_xyz_WF = np.matmul(lidar_xyz_RF, np.transpose(particle_rotmat_WF, (0, 2, 1))) + particle_origin_WF
    
    return lidar_xyz_WF


# def temp_add(a, b):
#     """
#         a: (num_particles, num_scan_ang, 2)
#         b: (9, 9, 2)
#     """
#     num_particles = a.shape[0]
#     num_scan_ang = a.shape[1]
#     grid_size = b.shape[0]
    
#     b1 = np.expand_dims(np.transpose(b, (2, 0, 1)), axis=0)
#     b2 = np.expand_dims(np.repeat(b1, num_scan_ang, axis=0), axis=0)
#     b3 = np.repeat(b2, num_particles, axis=0)

#     a1 = np.repeat(np.expand_dims(a, axis=3), grid_size, axis=3)
#     a2 = np.repeat(np.expand_dims(a1, axis=4), grid_size, axis=4)
    
#     return a2 + b3


# def compute_map_correlation(MAP, lidar_xyz_WF):
#     """
#         MAP: dict of current occupancy grid map
#         lidar_xyz_WF: (num_particles, 1081, 3)
#     """
# #     xy_grid_locs = np.array([np.arange(MAP['xmin'], MAP['xmax']+MAP['res'], MAP['res']), 
# #                              np.arange(MAP['ymin'], MAP['ymax']+MAP['res'], MAP['res'])])
# #     print("stage-1:", xy_grid_locs.shape)
    
#     xy_grid_9x9 = np.meshgrid(np.arange(-0.2, 0.2+MAP['res'], MAP['res']), 
#                               np.arange(-0.2, 0.2+MAP['res'], MAP['res']))   # (9,9), (9,9)
    
#     xy_grid_9x9 = np.stack((xy_grid_9x9[1], xy_grid_9x9[0]), axis=2)       # (9, 9, 2)

#     lidar_xyz_WF_9x9_phy_locs = temp_add(lidar_xyz_WF[:, :, 0:2], xy_grid_9x9)   # (num_particles, 1081, 2, 9, 9)
    
#     lidar_xyz_WF_9x9_grid_locs = np.int16(np.round((lidar_xyz_WF_9x9_phy_locs - MAP['xmin']) / MAP['res']))
    
#     valid_xloc = np.logical_and(lidar_xyz_WF_9x9_grid_locs[:,:,0] >= 0, lidar_xyz_WF_9x9_grid_locs[:,:,0] < MAP['sizex'])
#     valid_yloc = np.logical_and(lidar_xyz_WF_9x9_grid_locs[:,:,1] >= 0, lidar_xyz_WF_9x9_grid_locs[:,:,1] < MAP['sizey'])
    
#     valid_grid_locs = np.logical_and(valid_xloc, valid_yloc)      # (num_particles, 1081, 9, 9)
    
#     x_locs = lidar_xyz_WF_9x9_grid_locs[:,:,0,:,:]   # (num_particles, 1081, 9, 9)
#     y_locs = lidar_xyz_WF_9x9_grid_locs[:,:,1,:,:]   # (num_particles, 1081, 9, 9)

#     x_locs = np.multiply(x_locs, valid_grid_locs)
#     y_locs = np.multiply(y_locs, valid_grid_locs)

#     valid_map = MAP['map'][x_locs, y_locs]            # (num_particles, 1081, 9, 9)
    
#     corr_mat = np.sum(valid_map, axis=1)             # (num_particles, 9, 9)

#     return corr_mat


def compute_map_correlation(MAP, vp, xs, ys):
    '''
        INPUT 
        im              the map 
        x_im,y_im       physical x,y positions of the grid map cells
        vp[:,:,0:2]       occupied x,y positions from range sensor (in physical unit)  
        xs,ys           physical x,y,positions you want to evaluate "correlation" 

        OUTPUT 
        c               sum of the cell values of all the positions hit by range sensor
    '''
    MAP['map'][0,0] = 0
    num_particles = vp.shape[0]
    
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((num_particles, nxs, nys))
    
    for jy in range(0, nys):
        y1 = vp[:, :, 1] + ys[jy]      # num_particles x 1081
        iy = np.int16(np.round((y1 - MAP['ymin']) / MAP['res']))   # num_particles x 1081
        
        valid_y = np.logical_and(iy >= 0, iy < MAP['sizey'])  # (num_particles, 1081)
        
        for jx in range(0, nxs):
            x1 = vp[:, :, 0] + xs[jx]             # num_particles x 1081
            
            ix = np.int16(np.round((x1 - MAP['xmin']) / MAP['res']))  # num_particles x 1081
            
            valid_x = np.logical_and(ix >= 0, ix < MAP['sizex'])  # num_particles x 1081
            valid = np.logical_and(valid_x, valid_y)            # num_particles x 1081

            final_ix = np.multiply(ix, valid)
            final_iy = np.multiply(iy, valid)

            tmp2 = 1.0 - MAP['map'][final_ix, final_iy]      # (num_particles, 1081)
            cpr[:, jx, jy] = np.sum(tmp2, axis=1)
    
    return cpr


def PF_predict_update(encoder_stamps, lidar_xyz_RF, valid_lidar_scan, robot_vel, init_scan_map, init_scan_map_logodds, num_particles=100, update_skip=5, std_v=0.5, std_omega=0.05, ds_idx=20):

    MAP = get_empty_map()
    MAP['map'] = np.copy(init_scan_map)
    MAP['logodds'] = np.copy(init_scan_map_logodds)
    MAP['map'][0,0] = 0

    # xy_grid_9x9 = np.meshgrid(np.arange(-0.2, 0.2+MAP['res'], MAP['res']), 
    #                           np.arange(-0.2, 0.2+MAP['res'], MAP['res']))   # (9,9), (9,9)
    # xy_grid_9x9 = np.stack((xy_grid_9x9[1], xy_grid_9x9[0]), axis=2)       # (9, 9, 2)

    # initialize the particle states and weights
    particle_states = np.zeros((num_particles, 3))
    particle_weights = np.full((num_particles), 1./num_particles)

    tau_t = encoder_stamps[1:] - encoder_stamps[0:-1]     # (4955,)
    num_scan_ang = lidar_xyz_RF.shape[1]
    num_traj_steps = encoder_stamps.shape[0] - 1
    
    # initialize the optimal robot trajectory to be estimated
    opt_robot_traj = np.zeros((num_traj_steps+1, 3))

    xs = np.arange(-0.2, 0.2+0.05, 0.05)   # (9,)
    ys = np.arange(-0.2, 0.2+0.05, 0.05)   # (9,)

    for traj_idx in tqdm(range(0, num_traj_steps)):
        particle_states = motion_model(particle_states, robot_vel[traj_idx], tau_t[traj_idx], 
                                       std_lin_vel=std_v, std_ang_vel=std_omega)                      # (num_particles, 3)

        if traj_idx % update_skip == 0:
            # convert particle xytheta in WF to 4x4 SE(3) pose in WF
            particle_se3pose_WF = get_se3pose_from_xytheta(particle_states)   # (num_particles, 4, 4)

            # take the lidar xyz points and convert to world frame using each particle's pose
            lidar_xyz_WF = compute_lidar_xyz_WF_vect2(lidar_xyz_RF[traj_idx], particle_se3pose_WF)  # (num_particles, 1081, 3)

            # compute map correlation between lidar xyz points corresponding to each particle's pose
            corr_mat = compute_map_correlation(MAP, lidar_xyz_WF, xs, ys)      # (num_particles, 9, 9)

            # find the best correlation location for each particle
            corr_per_particle = np.max(np.max(corr_mat, axis=2), axis=1)      # (num_particles)

            # update the weights of each particle
            particle_weights = np.multiply(particle_weights, corr_per_particle) / np.dot(particle_weights, corr_per_particle)
            assert(np.abs(np.sum(particle_weights) - 1.) <= 1e-6)

        # find the particle with the highest weights after update
        best_particle_idx = np.argmax(particle_weights)

        # check if the effective number of particles is less than some threshold
        num_eff_particles = 1./np.dot(particle_weights, particle_weights)
        if num_eff_particles <= num_particles / 10:
            # if true, then resample the particle set (states, weights)
            # sample randomly from set {0,2,...,N-1} with particle weights as probability
            draw_set = np.random.choice(np.arange(0, num_particles), size=num_particles, p=particle_weights)
            particle_states = particle_states[draw_set, :]
            particle_weights = np.full((num_particles), 1./num_particles)

        # update the map using the best particle's pose
        best_particle_state = np.expand_dims(particle_states[best_particle_idx, :], axis=0)   # (1, 3)
        best_particle_se3pose_WF = get_se3pose_from_xytheta(best_particle_state)             # (1, 4, 4)
        
        # update the optimal robot trajectory array
        opt_robot_traj[traj_idx + 1, :] = np.copy(best_particle_state)

        # convert lidar scans from current time step to WF using best particle's se3pose
        lidar_xyz_WF_BP = compute_lidar_xyz_WF_vect2(lidar_xyz_RF[traj_idx], best_particle_se3pose_WF)  # (1, 1081, 3)

        # convert from meters to cell index
        ex_cells = np.ceil((lidar_xyz_WF_BP[0, :, 0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1  # (1081,)
        ey_cells = np.ceil((lidar_xyz_WF_BP[0, :, 1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1  # (1081,)

        sx = best_particle_state[0, 0] + LIDAR_SE3POSE_RF[0, 3]
        sy = best_particle_state[0, 1] + LIDAR_SE3POSE_RF[1, 3]
        sx_cells = np.ceil((sx - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
        sy_cells = np.ceil((sy - MAP['ymin']) / MAP['res']).astype(np.int16) - 1

        for scan_idx in range(num_scan_ang):
            if valid_lidar_scan[traj_idx, scan_idx] == False:
                continue
            
            ray_x, ray_y = bresenham2D(sx=sx_cells, sy=sy_cells, 
                                             ex=ex_cells[scan_idx], ey=ey_cells[scan_idx]).astype(np.int16)

            valid_xy = np.logical_and(np.logical_and(ray_x >= 0, ray_x < MAP['sizex']), 
                                      np.logical_and(ray_y >= 0, ray_y < MAP['sizey']))

            ray_x = ray_x[valid_xy]
            ray_y = ray_y[valid_xy]

            if ray_x.shape[0] == 0:
                continue

            # update the map logodds
            MAP['logodds'][ray_x[:-1], ray_y[:-1]] -= LOG4   # free
            MAP['logodds'][ray_x[-1], ray_y[-1]] += LOG4     # occupied

        # constrain log-odds map to avoid overconfident estimation
        MAP['logodds'] = np.clip(MAP['logodds'], a_min=-100, a_max=50)
        temp = 1. / (1 + np.exp(MAP['logodds']))
        MAP['map'][temp > 0.5] = 1.
        MAP['map'][temp < 0.5] = 0
        MAP['map'][temp == 0.5] = 0.5
        
        # save map with particles location after every few iterations
        if traj_idx % 500 == 0:
            savepath = "../plots/ds"+str(ds_idx)+"/map_pf_N"+str(num_particles)+"_stdv"+str(std_v)+"_stdw"+str(std_omega)+"_itr"+str(traj_idx)+".png"
            plot_grid_map_particles_traj(MAP['map'], traj=opt_robot_traj[0:traj_idx], p_states=particle_states, savepath=savepath)

    return MAP['map'], MAP['logodds'], opt_robot_traj


def get_kinect_se3pose_RF():
    phi = 0
    theta = 0.36
    psi = 0.021

    Rx_phi = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
    Ry_theta = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    Rz_psi = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])

    KINECT_ROTMAT_RF = np.matmul(Rz_psi, np.matmul(Ry_theta, Rx_phi))
    KINECT_ORIGIN_RF = np.array([332.76 - 330.2/2, 0, 380.01]) / 1000.0

    KINECT_SE3POSE_RF = np.eye(4)
    KINECT_SE3POSE_RF[0:3, 0:3] = KINECT_ROTMAT_RF
    KINECT_SE3POSE_RF[0:3, 3] = KINECT_ORIGIN_RF
    
    return KINECT_SE3POSE_RF


def plot_texture_map(texture_map, savepath):
    plt.figure(figsize=(10,10))
    plt.imshow(texture_map['map'])
    plt.ylabel("X (world-frame)", fontsize=14)
    plt.xlabel("Y (world-frame)", fontsize=14)
    plt.title("Texture map", fontsize=14)

    plt.savefig(savepath)
    plt.show()
    plt.close()
        

def gen_texture_map(robot_traj_WF, rgb_stamps, disp_stamps, encoder_stamps, num_particles, stdv, stdw, ds_idx):
    disp_dir = "../data/dataRGBD/Disparity"+str(ds_idx)+"/"
    rgb_dir = "../data/dataRGBD/RGB"+str(ds_idx)+"/"
    
    disp_match_rgb_idx = match_closest_stamps(rgb_stamps, disp_stamps)
    enc_match_rgb_idx = match_closest_stamps(rgb_stamps, encoder_stamps)
    
    robot_traj_WF_match = robot_traj_WF[enc_match_rgb_idx]
    robot_se3pose_WF = get_se3pose_from_xytheta(robot_traj_WF_match)    # (4956, 4, 4)
    
    WHL_RAD = 0.254/2    # (in meters)

    # get 3D coordinates 
    fu = 585.05108211
    fv = 585.05108211
    cu = 242.94140713
    cv = 315.83800193
    K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    KINECT_SE3POSE_RF = get_kinect_se3pose_RF()

    # create a second grid map for rgb colors
    texture_map = get_texture_map()

    # read images from the rgb dir and read corresponding disparity image
    for img_idx in tqdm(range(rgb_stamps.shape[0])):
    
        rgb_img_path = os.path.join(rgb_dir, "rgb"+str(ds_idx)+"_"+str(img_idx+1)+".png")
        rgb_img = cv2.imread(rgb_img_path)[...,::-1]                      # (480 x 640 x 3)

        corr_disp_img_idx = disp_match_rgb_idx[img_idx]
        disp_img_path = os.path.join(disp_dir, "disparity"+str(ds_idx)+"_"+str(corr_disp_img_idx+1)+".png")

        # convert disparity from uint16 to double
        disp_img = cv2.imread(disp_img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)           # (480 x 640)
        img_height, img_width = disp_img.shape

        # get depth image
        dd = -0.00304 * disp_img + 3.31
        depth_img = 1.03 / dd            # (480, 640)

        # calculate u and v coordinates 
        v, u = np.mgrid[0:img_height, 0:img_width]

        # calculate the location of each pixel in the RGB image
        rgbu = np.round((u * 526.37 + dd * (-4.5 * 1750.46) + 19276.0) / fu)  # (480, 640)
        rgbv = np.round((v * 526.37 + 16662.0) / fv)                          # (480, 640)
        valid_locs = (rgbu >= 0)&(rgbu < img_height)&(rgbv >= 0)&(rgbv < img_width)   # (480, 640)

        rgb_px_coords = np.zeros((img_height, img_width, 3))
        rgb_px_coords[:,:,0] = np.copy(rgbu)
        rgb_px_coords[:,:,1] = np.copy(rgbv)
        rgb_px_coords[:,:,2] = 1.0

        # convert from pixel coordinates to camera optical frame
        tmp1 = np.multiply(rgb_px_coords, np.repeat(np.expand_dims(depth_img, axis=2), 3, axis=2))  # (480, 640, 3)
        rgb_coords_OF = np.matmul(tmp1, Kinv.T)   # (480, 640, 3)

        # convert from camera optical frame to regular frame
        rot_mat_RGF2OF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])  # (3,3)
        rgb_coords_RGF = np.matmul(rgb_coords_OF, np.linalg.inv(rot_mat_RGF2OF).T)  # (480, 640, 3)

        # convert from regular frame to robot body frame
        rgb_coords_RF = np.matmul(rgb_coords_RGF, KINECT_SE3POSE_RF[0:3,0:3].T) + KINECT_SE3POSE_RF[0:3,3]   # (480, 640, 3)

        # convert from robot body frame to world frame
        robot_rotmat_WF = robot_se3pose_WF[img_idx, 0:3, 0:3]
        robot_origin_WF = robot_se3pose_WF[img_idx, 0:3, 3]

        rgb_coords_WF = np.matmul(rgb_coords_RF, robot_rotmat_WF.T) + robot_origin_WF   # (480, 640, 3)
        floor_plane = np.logical_and(rgb_coords_WF[:,:,2] >= WHL_RAD - 0.005, rgb_coords_WF[:,:,2] <= WHL_RAD + 0.005)  # (480, 640)
        
        rgb_coords_WF_cells = np.ceil((rgb_coords_WF[:,:,0:2] - MAP_XMIN) / MAP_RES).astype(np.int16) - 1  # (480, 640, 2)

        x_locs = rgb_coords_WF_cells[:,:,0]
        y_locs = rgb_coords_WF_cells[:,:,1]

        valid_locs_2 = np.logical_and(np.logical_and(x_locs >= 0, x_locs < texture_map['sizex']), 
                                      np.logical_and(x_locs >= 0, y_locs < texture_map['sizey']))    # (480, 640)

        final_valid_locs = np.logical_and(np.logical_and(valid_locs, valid_locs_2), floor_plane)

        texture_map['map'][x_locs[final_valid_locs], y_locs[final_valid_locs]] = rgb_img[rgbu[final_valid_locs].astype(int), 
                                                                                         rgbv[final_valid_locs].astype(int)] / 255.0
        # save map with particles location after every few iterations
        if img_idx % 400 == 0 and img_idx != 0:
            savepath = "../plots/ds"+str(ds_idx)+"/texture_map_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+"_itr"+str(img_idx)+".png"
            plot_texture_map(texture_map, savepath)


    return texture_map