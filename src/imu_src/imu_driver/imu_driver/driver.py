#import rospy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
import serial
from datetime import datetime
import numpy as np
import sys
from imu_msg.msg import IMUmsg
from imu_msg.msg import *
import time
import math

class IMUDriver(Node):
    def __init__(self):
        super().__init__('IMUDriver')
        self.declare_parameter('port', '/dev/pts/5')
        self.declare_parameter('baudrate', 115200)

        serial_port = self.get_parameter('port').value
        serial_baud = self.get_parameter('baudrate').value

        self.port = serial.Serial(serial_port, serial_baud, timeout=3.0)
        self.port.write(b"$VNWRG,07,40*59")
        self.publisher = self.create_publisher(IMUmsg, 'imu', 10)
        self.timer = self.create_timer(0.01, self.read_data)

    def read_data(self):
        data = self.port.readline().decode().strip()
        if "$VNYMR" in data:
            self.parse_imu(data)

    def parse_imu(self, data):
        try:
            parts = data.split(',')
            if parts[0] != "$VNYMR":
                return None
            yaw =  math.radians(float(parts[1]))
            pitch = math.radians(float(parts[2]))
            roll = math.radians(float(parts[3]))
            magX = float(parts[4])
            magY = float(parts[5])
            magZ = float(parts[6])
            accX = float(parts[7])
            accY = float(parts[8])
            accZ = float(parts[9])
            gyroX = float(parts[10])
            gyroY = float(parts[11])
            gyroZ = float(parts[12].split('*')[0])
            #convert to quaternion
            qw = np.cos(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) + np.sin(yaw/2) * np.sin(pitch/2) * np.sin(roll/2)
            qx = np.cos(yaw/2) * np.cos(pitch/2) * np.sin(roll/2) - np.sin(yaw/2) * np.sin(pitch/2) * np.cos(roll/2)
            qy = np.cos(yaw/2) * np.sin(pitch/2) * np.cos(roll/2) + np.sin(yaw/2) * np.cos(pitch/2) * np.sin(roll/2)
            qz = np.sin(yaw/2) * np.cos(pitch/2) * np.cos(roll/2) - np.cos(yaw/2) * np.sin(pitch/2) * np.sin(roll/2)

            now = self.get_clock().now()
            secs = now.seconds_nanoseconds()[0]
            nanosecs = now.seconds_nanoseconds()[1]
            msg = IMUmsg()
            msg.header.stamp.sec = secs
            msg.header.stamp.nanosec = nanosecs
            msg.header.frame_id = 'IMU1_Frame'
            msg.imu.orientation.x = qx
            msg.imu.orientation.y = qy
            msg.imu.orientation.z = qz
            msg.imu.orientation.w = qw
            msg.imu.linear_acceleration.x = accX
            msg.imu.linear_acceleration.y = accY
            msg.imu.linear_acceleration.z = accZ
            msg.imu.angular_velocity.x = gyroX
            msg.imu.angular_velocity.y = gyroY
            msg.imu.angular_velocity.z = gyroZ
            msg.mag_field.magnetic_field.x = magX*(1e-4)
            msg.mag_field.magnetic_field.y = magY*(1e-4)
            msg.mag_field.magnetic_field.z = magZ*(1e-4)
            msg.my_string = data

            self.publisher.publish(msg)
            self.get_logger().info(f'Current IMU status: {msg}')
        except(ValueError, IndexError) as e:
            self.get_logger().warn(f"Invalid Data: {e}")
        except serial.SerialException as e:
            self.get_logger().error(f"Serial communication error: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error: {e}")

def main(args=None):
    rclpy.init(args=args)
    imu_driver = IMUDriver()

    try:
        rclpy.spin(imu_driver)
    except KeyboardInterrupt:
        pass
    finally:
        imu_driver.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()