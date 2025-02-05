import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
import math

#Magnetometer Calibration

file_path = '/home/atharva/Downloads/imu_data_circles.csv'  
data = pd.read_csv(file_path)


data['time'] = data['time_sec'] + data['time_nanosec'] / 1e9


start_time = 1729525384
end_time = 1729525422
data = data[(data['time_sec'] > start_time) & (data['time_sec'] < end_time)]


magnetic_x = data['mag_x'].to_numpy()
magnetic_y = data['mag_y'].to_numpy()


x_offset = (magnetic_x.min() + magnetic_x.max()) / 2.0
y_offset = (magnetic_y.min() + magnetic_y.max()) / 2.0
calibrated_x = magnetic_x - x_offset
calibrated_y = magnetic_y - y_offset


def ellipse_fit(params, x, y):
    x0, y0, a, b, theta = params
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_trans = cos_theta * (x - x0) + sin_theta * (y - y0)
    y_trans = -sin_theta * (x - x0) + cos_theta * (y - y0)
    return (x_trans / a) ** 2 + (y_trans / b) ** 2 - 1


initial_guess = [0, 0, np.std(calibrated_x), np.std(calibrated_y), 0]


params, _ = leastsq(ellipse_fit, initial_guess, args=(calibrated_x, calibrated_y))
x0, y0, a, b, theta = params


cos_theta, sin_theta = np.cos(theta), np.sin(theta)
R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


soft_iron_calibrated = np.dot(R, np.array([calibrated_x - x0, calibrated_y - y0]))


raw_radius_mean = np.mean(np.sqrt(magnetic_x**2 + magnetic_y**2))
calibrated_radius_mean = np.mean(np.sqrt(soft_iron_calibrated[0]**2 + soft_iron_calibrated[1]**2))
scaling_factor = raw_radius_mean / calibrated_radius_mean


final_calibrated_x, final_calibrated_y = scaling_factor * soft_iron_calibrated



hard_iron_radius_mean = np.mean(np.sqrt(calibrated_x**2 + calibrated_y**2))
soft_iron_radius_mean = np.mean(np.sqrt(final_calibrated_x**2 + final_calibrated_y**2))


normalization_scaling_factor = hard_iron_radius_mean / soft_iron_radius_mean


normalized_final_calibrated_x = final_calibrated_x * normalization_scaling_factor
normalized_final_calibrated_y = final_calibrated_y * normalization_scaling_factor

plt.figure(figsize=(24, 6))

plt.subplot(1, 4, 1)
plt.plot(magnetic_x, magnetic_y, label='Raw Data', color='gray', marker='o', markersize=2, alpha=0.5)
plt.plot(calibrated_x, calibrated_y, label='Hard-Iron Calibrated', color='blue', marker='x', markersize=2)
plt.title('Raw vs Hard-Iron Calibrated')
plt.xlabel('Magnetic X (Gauss)')
plt.ylabel('Magnetic Y (Gauss)')
plt.legend()
plt.grid()
plt.axis('equal')


plt.subplot(1, 4, 2)
plt.plot(magnetic_x, magnetic_y, label='Raw Data', color='gray', marker='o', markersize=2, alpha=0.5)
plt.plot(normalized_final_calibrated_x, normalized_final_calibrated_y, label='Normalized Soft-Iron Calibrated', color='green', marker='+', markersize=2)
plt.title('Raw vs Normalized Soft-Iron Calibrated')
plt.xlabel('Magnetic X (Gauss)')
plt.ylabel('Magnetic Y (Gauss)')
plt.legend()
plt.grid()
plt.axis('equal')


plt.subplot(1, 4, 3)
plt.plot(magnetic_x, magnetic_y, label='Raw Data', color='gray', marker='o', markersize=2, alpha=0.5)
plt.plot(normalized_final_calibrated_x, normalized_final_calibrated_y, label='Both Calibrated', color='purple', marker='x', markersize=2)
plt.title('Raw vs Both Hard-Iron and Normalized Soft-Iron Calibrated')
plt.xlabel('Magnetic X (Gauss)')
plt.ylabel('Magnetic Y (Gauss)')
plt.legend()
plt.grid()
plt.axis('equal')


plt.subplot(1, 4, 4)
plt.plot(calibrated_x, calibrated_y, label='Hard-Iron Calibrated', color='blue', marker='x', markersize=2)
plt.plot(normalized_final_calibrated_x, normalized_final_calibrated_y, label='Normalized Soft-Iron Calibrated', color='green', marker='+', markersize=2)
plt.title('Hard-Iron vs Normalized Soft-Iron Calibrated')
plt.xlabel('Magnetic X (Gauss)')
plt.ylabel('Magnetic Y (Gauss)')
plt.legend()
plt.grid()
plt.axis('equal')
plt.tight_layout()
plt.show()
	

 #Estimation of Yaw Angle

from scipy.integrate import cumtrapz
from scipy.signal import butter, filtfilt



data_path = '/home/atharva/Downloads/imu_data_driving.csv'
data = pd.read_csv(data_path)


data['time'] = data['time_sec'] + data['time_nanosec'] / 1e9
imu_time = data['time']

magnetic_x = data['mag_x'].to_numpy()
magnetic_y = data['mag_y'].to_numpy()


x_offset = (magnetic_x.min() + magnetic_x.max()) / 2.0
y_offset = (magnetic_y.min() + magnetic_y.max()) / 2.0
calibrated_x = magnetic_x - x_offset
calibrated_y = magnetic_y - y_offset


def ellipse_fit(params, x, y):
    x0, y0, a, b, theta = params
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_trans = cos_theta * (x - x0) + sin_theta * (y - y0)
    y_trans = -sin_theta * (x - x0) + cos_theta * (y - y0)
    return (x_trans / a) ** 2 + (y_trans / b) ** 2 - 1


initial_guess = [0, 0, np.std(calibrated_x), np.std(calibrated_y), 0]


params, _ = leastsq(ellipse_fit, initial_guess, args=(calibrated_x, calibrated_y))
x0, y0, a, b, theta = params


cos_theta, sin_theta = np.cos(theta), np.sin(theta)
R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


soft_iron_calibrated = np.dot(R, np.array([calibrated_x - x0, calibrated_y - y0]))


raw_radius_mean = np.mean(np.sqrt(magnetic_x**2 + magnetic_y**2))
calibrated_radius_mean = np.mean(np.sqrt(soft_iron_calibrated[0]**2 + soft_iron_calibrated[1]**2))
scaling_factor = raw_radius_mean / calibrated_radius_mean


final_calibrated_x, final_calibrated_y = scaling_factor * soft_iron_calibrated


hard_iron_radius_mean = np.mean(np.sqrt(calibrated_x**2 + calibrated_y**2))
soft_iron_radius_mean = np.mean(np.sqrt(final_calibrated_x**2 + final_calibrated_y**2))


normalization_scaling_factor = hard_iron_radius_mean / soft_iron_radius_mean


normalized_final_calibrated_x = final_calibrated_x * normalization_scaling_factor
normalized_final_calibrated_y = final_calibrated_y * normalization_scaling_factor



time_series = imu_time.to_numpy()
raw_yaw = np.unwrap(np.degrees(np.arctan2(magnetic_x, magnetic_y)))
corrected_yaw = np.unwrap(np.degrees(np.arctan2(normalized_final_calibrated_x, normalized_final_calibrated_y))) -100


plt.figure(figsize=(14, 7))
plt.plot(raw_yaw, label="Raw Yaw Angle")
plt.plot(corrected_yaw, label="Corrected Yaw Angle")
plt.xlabel("Sample Index")
plt.ylabel("Yaw Angle (degrees)")
plt.title("Comparison of Raw and Corrected Yaw Angles")
plt.legend()
plt.grid(True)
plt.show()

w, x, y, z = data['q_w'], data['q_x'], data['q_y'], data['q_z']

t3 = +2.0 * (w * z + x * y)
t4 = +1.0 - 2.0 * (y * y + z * z)
yaw_imu = np.degrees(np.unwrap(np.arctan2(t3, t4)))

angv_z = data['av_z']
yaw_gyro = cumtrapz(angv_z, time_series, initial=0) * (180 / np.pi)
yaw_gyro_unwrap = np.unwrap(yaw_gyro)


plt.figure(figsize=(14, 7))
plt.plot(corrected_yaw, label="Magnetometer Yaw")
plt.plot(yaw_gyro_unwrap, label="Yaw Integrated from Gyro")
plt.xlabel("Sample Index")
plt.ylabel("Yaw Angle (degrees)")
plt.title("Comparison of Yaw Angles: Magnetometer vs. Integrated Gyro")
plt.legend()
plt.grid(True)
plt.show()


lpf = filtfilt(*butter(3, 0.1, "lowpass", fs=40), corrected_yaw)
hpf = filtfilt(*butter(3, 0.0001, 'highpass', fs=40), yaw_gyro_unwrap)


alpha = 0.75
complementary_yaw = alpha * hpf + (1 - alpha) * lpf


plt.plot(time_series, lpf, label='Low-Pass Filtered Yaw (Magnetometer)', color='red', alpha=0.75)
plt.plot(time_series, hpf, label='High-Pass Filtered Yaw (Gyro)', color='blue', alpha=0.75)
plt.plot(time_series, complementary_yaw, label='Complementary Filtered Yaw', color='orange', alpha=0.75)
plt.xlabel('Time (seconds)')
plt.ylabel('Yaw Angle (degrees)')
plt.title('Yaw Angle Comparison: Low-Pass Filter (Magnetometer), High-Pass Filter (Gyro) & Complementary Filter')
plt.legend(loc='upper left', fontsize='large', frameon=False)
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()


plt.plot(time_series, complementary_yaw, label='Complementary Filtered Yaw', color='orange', alpha=0.75)
plt.plot(time_series, yaw_imu, label="Yaw from IMU")
plt.xlabel('Time (seconds)')
plt.ylabel('Yaw Angle (degrees)')
plt.title('Yaw Angle Comparison: IMU Yaw & Complementary Filter')
plt.legend(loc='upper left', fontsize='large', frameon=False)
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()



#Estimation of forward velocity


imu_data = pd.read_csv('/home/atharva/Downloads/imu_data_driving.csv')
gps_data = pd.read_csv('/home/atharva/Downloads/gps_data_driving.csv')


imu_data['time'] = imu_data['time_sec'] + imu_data['time_nanosec'] / 1e9
gps_data['time'] = gps_data['time_sec'] + gps_data['time_nanosec'] / 1e9


linear_acc_x = imu_data['la_x'].to_numpy()
imu_time = imu_data['time'].to_numpy()


linear_acc_x -= np.mean(linear_acc_x)


forward_velocity_imu = cumtrapz(linear_acc_x, imu_time, initial=0)
forward_velocity_imu_wrap = np.unwrap(forward_velocity_imu)


utm_easting = gps_data['utm_easting'].to_numpy()
utm_northing = gps_data['utm_northing'].to_numpy()
gps_time = gps_data['time'].to_numpy()


distance = np.sqrt(np.diff(utm_easting)**2 + np.diff(utm_northing)**2)
gps_velocity = distance / np.diff(gps_time)


gps_time = gps_time[1:]


plt.figure(figsize=(12, 6))
plt.plot(imu_time, forward_velocity_imu_wrap, label='IMU-based Velocity (Original)', color='blue')
plt.plot(gps_time, gps_velocity, label='GPS-based Velocity', color='orange')
plt.title('Forward Velocity from IMU and GPS')
plt.xlabel('Time (time_sec)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.show()



adjusted_velocity_imu = np.copy(forward_velocity_imu_wrap)
adjusted_velocity_imu[adjusted_velocity_imu < 0] = 0


plt.figure(figsize=(12, 6))
plt.plot(imu_time, adjusted_velocity_imu, label='Corrected IMU-based Velocity (No Filter)', color='green')
plt.plot(gps_time, gps_velocity, label='GPS-based Velocity', color='orange')
plt.title('Corrected IMU and GPS Velocity Comparison')
plt.xlabel('Time (time_sec)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.show()


#Dead Reckoning

imu_data = pd.read_csv('/home/atharva/Downloads/imu_data_driving.csv')
gps_data = pd.read_csv('/home/atharva/Downloads/gps_data_driving.csv')


imu_time = imu_data['time_sec']
gps_time = gps_data['time_sec']
mag_z = imu_data['mag_z']
mag_x = imu_data['mag_x']
mag_y = imu_data['mag_y']
gyro_z = imu_data['av_z']
accel_x = imu_data['la_x']
accel_y = imu_data['la_y']
utm_easting = np.array(gps_data['utm_easting'])  
utm_northing = np.array(gps_data['utm_northing'])  
w, x, y, z = imu_data['q_w'], imu_data['q_x'], imu_data['q_y'], imu_data['q_z']


t3 = +2.0 * (w * z + x * y)
t4 = +1.0 - 2.0 * (y * y + z * z)
yaw_i = np.degrees(np.unwrap(np.arctan2(t3, t4)))


imu_time = imu_time - imu_time[0]
gps_time = gps_time - gps_time[0]


yaw_imu = np.unwrap(yaw_i)


x_offset= (mag_x.max() + mag_x.min()) / 2
y_offset = (mag_y.max() + mag_y.min()) / 2
print(f'hard iron offset x = {x_offset}, y = {y_offset}')
mag_hi_x = mag_x - x_offset
mag_hi_y= mag_y - y_offset


rad_x = (mag_hi_x.max() - mag_hi_x.min()) / 2
rad_y = (mag_hi_y.max() - mag_hi_y.min()) / 2
rad = math.sqrt(rad_x**2 + rad_y**2)
theta = math.atan2(rad_y, rad_x)


rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
rot_data = np.matmul(rot_mat, np.vstack((mag_hi_x, mag_hi_y)))


des_rad = 0.2
x_scale = des_rad / rad_x
y_scale= des_rad / rad_y
scale_mat = np.array([[x_scale, 0], [0, y_scale]])
scale_data = np.matmul(scale_mat, rot_data)


rev_rot = np.array([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]])
final_data = np.matmul(rev_rot, scale_data)


yaw = np.arctan2(mag_y, mag_x)
yaw_deg = np.degrees(np.unwrap(yaw))
yaw_calibrated = np.arctan2(final_data[0], final_data[1])
yaw_calibrated_deg = np.degrees(np.unwrap(yaw_calibrated))
yaw_angle = cumtrapz(gyro_z, imu_time, initial=0)
yaw_angle_deg = np.degrees(np.unwrap(yaw_angle))


alpha = 0.9
def lpf(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def hpf(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

fs = 1 / np.mean(np.diff(imu_time))
cutoff_lpf = 0.1
yaw_calibrated_deg_f = lpf(yaw_calibrated_deg, cutoff=cutoff_lpf, fs=fs)
cutoff_hpf = 0.0001
yaw_angle_deg_f = hpf(yaw_angle_deg, cutoff=cutoff_hpf, fs=fs)
yaw_complementary = alpha * yaw_angle_deg + (1 - alpha) * yaw_calibrated_deg


vel_ac = cumtrapz(accel_x, imu_time, initial=0)


time_diff = np.diff(gps_time).astype(float)  
time_diff[time_diff == 0] = np.nan  
vel_gps = np.sqrt(np.diff(utm_easting)**2 + np.diff(utm_northing)**2) / time_diff
vel_gps = np.insert(vel_gps, 0, 0)
vel_gps = np.nan_to_num(vel_gps)  

ac_c = accel_x - np.mean(accel_x)
vel_ac_c = cumtrapz(ac_c, imu_time, initial=0)
vel_ac_c[vel_ac_c < 0] = 0
mask = (vel_gps > 0) & (vel_gps < 0.18)
vel_ac_c[np.isin(imu_time, gps_time[mask])] = 0


disp = cumtrapz(vel_ac_c, initial=0)
x2dot = accel_x
x1dot = cumtrapz(accel_x, imu_time, initial=0)
y2dot = gyro_z * x1dot
Y_obs = accel_y
fv = vel_ac_c
yc = yaw_complementary
ve = fv * np.sin(np.radians(yaw_imu - 19))
vn = fv * np.cos(np.radians(yaw_imu - 19))
xe = cumtrapz(ve, imu_time, initial=0)
xn = cumtrapz(vn, imu_time, initial=0)


scaling = 0.70
xe_s = xe * scaling
xn_s = xn * scaling
xe_a = xe_s - xe_s[0] + utm_easting[0]
xn_a = xn_s - xn_s[0] + utm_northing[0]


larger_correction_angle = np.radians(6)
larger_correction_matrix = np.array([[np.cos(larger_correction_angle), -np.sin(larger_correction_angle)],
                                     [np.sin(larger_correction_angle), np.cos(larger_correction_angle)]])


corrected_trajectory_larger = np.matmul(larger_correction_matrix, np.vstack((xe_s, xn_s)))
xe_a_corrected_larger = corrected_trajectory_larger[0] - corrected_trajectory_larger[0][0] + utm_easting[0]
xn_a_corrected_larger = corrected_trajectory_larger[1] - corrected_trajectory_larger[1][0] + utm_northing[0]


plt.figure(figsize=(10, 6))
plt.plot(utm_easting, utm_northing, label='GPS track', color='green')
plt.plot(xe_a_corrected_larger, xn_a_corrected_larger, label='Estimated track', color='red')
plt.xlabel('Eastward Position (m)')
plt.ylabel('Northward Position (m)')
plt.title('Comparison of GPS and Estimated Trajectory')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 6))
plt.plot(xe_a, xn_a, label='Estimated track', color='green')
plt.xlabel('xe (m)')
plt.ylabel('xn (m)')
plt.title('Estimated Trajectory')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 10))
plt.plot(Y_obs, label='Y Observed', color='blue')
plt.plot(y2dot / -1, label='wX(dot)', color='red')
plt.legend(loc='upper right', fontsize='x-large')
plt.grid(color='grey', linestyle='--', linewidth=1)
plt.title('Y Observed vs wX(dot)')
plt.xlabel('Samples @40Hz')
plt.ylabel('Acceleration (m/s²)')
plt.show()


