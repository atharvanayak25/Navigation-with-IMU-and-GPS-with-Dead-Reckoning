from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # Define the launch description with the port argument and node
    return LaunchDescription([
        Node(
                package='gps_driver',
                executable='driver',
                name='gps_driver',
                output='screen',
                parameters=[{'port': LaunchConfiguration('port')}],
            )
    ])
