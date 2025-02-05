import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


imu_data_path = '/home/atharva/Downloads/imu_data_driving.csv'
gps_data_path = '/home/atharva/Downloads/gps_data_driving.csv'
imu_data = pd.read_csv(imu_data_path)
gps_data = pd.read_csv(gps_data_path)


imu_data['time'] = imu_data['time_sec'] + imu_data['time_nanosec'] / 1e9
gps_data['time'] = gps_data['time_sec'] + gps_data['time_nanosec'] / 1e9
time = imu_data['time'].to_numpy()


a_x = imu_data['la_x'].to_numpy()       
omega = imu_data['av_z'].to_numpy()     


if omega.max() > 2 * np.pi:  
    omega = np.radians(omega)  


V_x = cumtrapz(a_x, time, initial=0)    


dV_x = np.gradient(V_x, time)           


omega_filtered = np.where(np.abs(omega) < 0.1, np.nan, omega)  


x_c = (a_x - dV_x) / (omega_filtered**2)


x_c_estimate = np.nanmean(x_c)  
print("Estimated x_c:", x_c_estimate)


plt.figure(figsize=(10, 6))
plt.plot(time, x_c, label='x_c Estimate Over Time')
plt.axhline(y=x_c_estimate, color='r', linestyle='--', label=f'Mean x_c = {x_c_estimate:.2f}')
plt.xlabel('Time (s)')
plt.ylabel('x_c (m)')
plt.title('Estimation of x_c Over Time (Using X-Components Only)')
plt.legend()
plt.grid(True)
plt.show()


plt.tight_layout()
plt.show()
