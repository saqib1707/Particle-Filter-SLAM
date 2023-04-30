import numpy as np


def load_encoder_data(dataset_idx):
    data = np.load("../data/Encoders%d.npz"%dataset_idx)

    encoder_counts = data["counts"].T     # n x 4 encoder counts [FR, FL, RR, RL]
    encoder_stamps = data["time_stamps"]  # encoder time stamps
    print("Number of encoder samples:", encoder_stamps.shape[0])
    
    return encoder_counts, encoder_stamps


def load_lidar_data(dataset_idx):
    data = np.load("../data/Hokuyo%d.npz"%dataset_idx)

    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"][0,0] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"].T       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans
    
    print("Number of LIDAR samples:", lidar_stamps.shape[0])
    print("LIDAR FOV stats (rad): min_angle = %f, max_angle = %f, angle_increment = %f"%(lidar_angle_min, lidar_angle_max, 
                                                                                       lidar_angle_increment))
    print("LIDAR FOV stats (deg): min_angle = %f, max_angle = %f, angle_increment = %f"%(lidar_angle_min * 180 / np.pi, 
                                                                                   lidar_angle_max * 180 / np.pi, 
                                                                                   lidar_angle_increment * 180 / np.pi))

    print("LIDAR range stats (expected): min_range = %f, max_range = %f"%(lidar_range_min, lidar_range_max))
    print("LIDAR range stats (given)   : min_range = %f, max_range = %f"%(lidar_ranges.min(), lidar_ranges.max()))
    
    return lidar_angle_min, lidar_angle_max, lidar_angle_increment, lidar_range_min, lidar_range_max, lidar_ranges, lidar_stamps


def load_IMU_data(dataset_idx):
    data = np.load("../data/Imu%d.npz"%dataset_idx)

    imu_angular_velocity = data["angular_velocity"].T # angular velocity in rad/sec
    imu_linear_accln = data["linear_acceleration"].T # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
    print("Number of IMU samples:", imu_stamps.shape[0])

    return imu_angular_velocity, imu_linear_accln, imu_stamps


def load_kinectRGBD_data(dataset_idx):
    data = np.load("../data/Kinect%d.npz"%dataset_idx)

    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images
    
    print("Number of RGB images = %d, Disparity images = %d"%(rgb_stamps.shape[0], disp_stamps.shape[0]))

    return disp_stamps, rgb_stamps


if __name__ == '__main__':
    dataset_idx = 20