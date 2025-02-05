from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare separate port arguments for GPS and IMU drivers
    gps_port_arg = DeclareLaunchArgument(
        'gps_port',
        default_value='/dev/ttyUSB0',  # Default GPS port
        description='Serial port for the GPS driver'
    )
    
    imu_port_arg = DeclareLaunchArgument(
        'imu_port',
        default_value='/dev/ttyUSB1',  # Default IMU port
        description='Serial port for the IMU driver'
    )

    # Get the directory where the launch files for the individual drivers are located
    gps_launch_dir = os.path.join(get_package_share_directory('gps_driver'), 'launch')
    imu_launch_dir = os.path.join(get_package_share_directory('imu_driver'), 'launch')

    # Include the launch files for both gps_driver and imu_driver, passing different ports
    gps_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gps_launch_dir, 'gps_launch.py')),
        launch_arguments={'port': LaunchConfiguration('gps_port')}.items()
    )
    
    imu_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(imu_launch_dir, 'imu_launch.py')),
        launch_arguments={'port': LaunchConfiguration('imu_port')}.items()
    )

    return LaunchDescription([
        gps_port_arg,
        imu_port_arg,
        gps_driver_launch,
        imu_driver_launch
    ])
