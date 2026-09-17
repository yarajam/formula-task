from launch import LaunchDescription
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    params_file = os.path.join(
        get_package_share_directory('perception_package'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='perception_package',
            executable='camera_node',
            name='camera_node',
            output='screen'
        ),
        Node(
            package='perception_package',
            executable='lidar_node',
            name='lidar_node',
            output='screen',
            parameters=[params_file]
        ),
        Node(
            package='perception_package',
            executable='visualization_node',
            name='visualization_node',
            output='screen'
        ),
    ])