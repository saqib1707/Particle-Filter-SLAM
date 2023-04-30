import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import argparse

import load_data
# import pr2_utils as utils
from slam_utils import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ds_idx', default=20, type=int)
    parser.add_argument('--tosave', default=False, action='store_true')
    parser.add_argument('--gen_map_dead_reck', default=False, action='store_true')
    parser.add_argument('--texture_map', default=False, action='store_true')
    parser.add_argument('--pf_update', default=False, action='store_true')
    parser.add_argument('--num_part', default=100, type=int)
    parser.add_argument('--stdv', default=0.5, type=float)
    parser.add_argument('--stdw', default=0.1, type=float)

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    
    encoder_counts, encoder_stamps = load_data.load_encoder_data(args.ds_idx)
    imu_ang_vel, imu_lin_accln, imu_stamps = load_data.load_IMU_data(args.ds_idx)
    lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamps = load_data.load_lidar_data(args.ds_idx)
    disp_stamps, rgb_stamps = load_data.load_kinectRGBD_data(args.ds_idx)

    imu_match_enc_idx = match_closest_stamps(encoder_stamps, imu_stamps)
    lidar_match_enc_idx = match_closest_stamps(encoder_stamps, lidar_stamps)
    assert(imu_match_enc_idx.shape[0] == lidar_match_enc_idx.shape[0] == encoder_stamps.shape[0])

    valid_lidar_scan = get_valid_lidar_scans(lidar_ranges)[lidar_match_enc_idx]   # (4956, 1081)
    
    # convert scan/range measurements from distances (in m) to (x,y,z) coordinates in LIDAR sensor frame
    lidar_xyz_LF = compute_lidar_xyz_LF(lidar_ranges, lidar_angle_min, lidar_angle_max, lidar_angle_increment)

    # convert from lidar sensor frame to robot body frame
    lidar_xyz_RF = compute_lidar_xyz_RF(lidar_xyz_LF)  # (4962, 1081, 3)
    lidar_xyz_RF = lidar_xyz_RF[lidar_match_enc_idx]              # (4956, 1081, 3)
    
    # ---------------------------------------------------------------------------------------------
    # Estimate LIDAR range measurements in world frame for the first LIDAR scan
    robot_pose_WF = np.eye(4)
    lidar_xyz_WF_init_scan = compute_lidar_xyz_WF(lidar_xyz_RF[0], robot_pose_WF)
    
    print("\nGenerating initial scan map...")
    # create an occupancy grid map of certain resolution
    init_scan_map, init_scan_map_logodds = gen_map_init_scan(lidar_xyz_WF_init_scan, valid_lidar_scan[0])

    if args.tosave == True:
        print("\nSaving the initial scan map...")
        savepath = "../plots/ds"+str(args.ds_idx)+"/init_scan_map.png"
        plot_grid_map(init_scan_map, init_scan_map_logodds, savepath)
    
    # ---------------------------------------------------------------------------------------------
    # Predict robot trajectory with dead-reckoning (no noise) using differential drive motion model starting from (0,0,0) pose
    robot_lin_vel, robot_ang_vel = compute_robot_vel(encoder_counts, encoder_stamps, imu_ang_vel)

    if args.tosave == True:
        print("\nSaving plots for robot linear and angular velocity with dead-reckoning...")
        plt.figure(figsize=(14,4))
        plt.subplot(1,2,1)
        plt.plot(encoder_stamps, robot_lin_vel)
        plt.xlabel("time", fontsize=14)
        plt.ylabel("linear velocity (in m/s)", fontsize=14)
        plt.title("Linear velocity measured by Encoder", fontsize=14)
        plt.grid(linestyle='--')

        plt.subplot(1,2,2)
        plt.plot(imu_stamps, robot_ang_vel)
        plt.xlabel("time", fontsize=14)
        plt.ylabel("angular velocity (in rad/s)", fontsize=14)
        plt.title("Angular velocity (yaw rate) measured by IMU", fontsize=14)
        plt.grid(linestyle='--')

        plt.savefig("../plots/ds"+str(args.ds_idx)+"/robot_vel.png")
        plt.show()
        plt.close()
        
    # Predict robot motion with dead-reckoning (no noise)
    robot_ang_vel_match = robot_ang_vel[imu_match_enc_idx]                  # (4956,)
    robot_vel = np.array([robot_lin_vel, robot_ang_vel_match]).T            # (4956, 2)

    # ---------------------------------------------------------------------------------------------
    # predict a single particle trajectory with no noise using motion model
    num_particles = 1
    stdv=0
    stdw=0
    savepath = "../plots/ds"+str(args.ds_idx)+"/traj_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".png"
    traj_dead_reck = pred_particle_traj(robot_vel, encoder_stamps, num_particles=num_particles, 
                                        std_lin_vel=stdv, std_ang_vel=stdw, tosave=args.tosave, savepath=savepath)
    
    # ---------------------------------------------------------------------------------------------
    # predict particle trajectories with some noise using motion model
    num_particles = 10
    stdv=0.1
    stdw=0.01
    savepath = "../plots/ds"+str(args.ds_idx)+"/traj_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".png"
    traj_mult_particles = pred_particle_traj(robot_vel, encoder_stamps, num_particles=num_particles, 
                                            std_lin_vel=stdv, std_ang_vel=stdw, tosave=args.tosave, savepath=savepath)
    
    # ---------------------------------------------------------------------------------------------
    # predict particle trajectories with some noise using motion model
    num_particles = 10
    stdv = 0.5
    stdw = 0.05
    savepath = "../plots/ds"+str(args.ds_idx)+"/traj_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".png"
    traj_mult_particles = pred_particle_traj(robot_vel, encoder_stamps, num_particles=num_particles, 
                                            std_lin_vel=stdv, std_ang_vel=stdw, tosave=args.tosave, savepath=savepath)
    
    # ---------------------------------------------------------------------------------------------
    # predict particle trajectories with some noise using motion model
    num_particles = 10
    stdv = 1.0
    stdw = 0.1
    savepath = "../plots/ds"+str(args.ds_idx)+"/traj_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".png"
    traj_mult_particles = pred_particle_traj(robot_vel, encoder_stamps, num_particles=num_particles, 
                                            std_lin_vel=stdv, std_ang_vel=stdw, tosave=args.tosave, savepath=savepath)

    # ---------------------------------------------------------------------------------------------
    # Generate occupancy grid map for robot trajectory with dead-reckoning
    if args.gen_map_dead_reck == True:
        print("\nGenerate full map with dead-reckoning and no noise...")
        # predict a single particle trajectory with no noise using motion model
        num_particles = 1
        traj_dead_reck_WF = pred_particle_traj(robot_vel, encoder_stamps, num_particles=num_particles, 
                                               std_lin_vel=0, std_ang_vel=0)     # (num_particles, 4956, 3)

        map_dead_reck, map_dead_reck_logodds = gen_map_dead_reck(lidar_xyz_RF, traj_dead_reck_WF[0], valid_lidar_scan)

        savepath = "../plots/ds"+str(args.ds_idx)+"/map_dead_reck.png"
        plot_grid_map(map_dead_reck, map_dead_reck_logodds, savepath)
        
        savepath = "../plots/ds"+str(args.ds_idx)+"/map_dead_reck_traj.png"
        plot_grid_map_particles_traj(map_dead_reck, traj=traj_dead_reck_WF, savepath=savepath)
        
    # --------------------------------------------------------------------------------------------- 
    # Predict - Update Particle Filter
    if args.pf_update == True:
        print("\nParticle-Filter Prediction-Update and Map Update...")
        num_particles = args.num_part
        stdv = args.stdv
        stdw = args.stdw

        map_pf, map_pf_logodds, opt_robot_traj = PF_predict_update(encoder_stamps, lidar_xyz_RF, valid_lidar_scan, robot_vel, init_scan_map, init_scan_map_logodds, num_particles=num_particles, update_skip=5, std_v=stdv, std_omega=stdw, ds_idx=args.ds_idx)
        
        savepath = "../plots/ds"+str(args.ds_idx)+"/map_pf_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".png"
        plot_grid_map(map_pf, map_pf_logodds, savepath)

        savepath = "../plots/ds"+str(args.ds_idx)+"/map_pf_traj_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".png"
        plot_grid_map_particles_traj(map_pf, traj=opt_robot_traj, savepath=savepath)
        
        savepath = "../data/gen_data/ds"+str(args.ds_idx)+"/opt_robot_traj_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".npy"
        np.save(savepath, opt_robot_traj)

    # ---------------------------------------------------------------------------------------------
    if args.texture_map == True:
        print("\nGenerate texture map using optimal robot trajectory...")
        # predict a single particle trajectory with no noise using motion model
        num_particles = args.num_part
        stdv = args.stdv
        stdw = args.stdw

        loadpath = "../data/gen_data/ds"+str(args.ds_idx)+"/opt_robot_traj_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".npy"
        opt_robot_traj = np.load(loadpath)
        texture_map = gen_texture_map(opt_robot_traj, rgb_stamps, disp_stamps, encoder_stamps, num_particles, stdv, stdw, args.ds_idx)
        
        savepath = "../plots/ds"+str(args.ds_idx)+"/texture_map_N"+str(num_particles)+"_stdv"+str(stdv)+"_stdw"+str(stdw)+".png"
        plot_texture_map(texture_map, savepath)