
import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from tf2_ros import Buffer, TransformListener, TransformException

from core.transformation import transform_points
from core.roi import extract_roi
from core.ground_removal import remove_ground
from core.clustering import euclidean_clustering

"""
Node: LiDAR Perception

Input:
    /velodyne_points

Outputs:
    /perception/roi_points
    /perception/non_ground_points
    /perception/clustered_points

Coordinate Frames:
    Source:
        The frame provided by the incoming PointCloud2 message
        (msg.header.frame_id).

    Target:
        base_footprint

The LiDAR points should be transformed from the sensor frame into
base_footprint before processing so that the coordinates are aligned
with the vehicle.

If this is unfamiliar, briefly read about ROS 2 TF2 and coordinate
frame transformations.
"""


class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        # self.target_frame = 'base_footprint'
        self.frame_logged = False #do i know the frame of the incoming point cloud? if not, log it once and then set this to True
        self.transform_logged = False #do i know the transform i need from the source frame to the target frame? if not, log it once and then set this to True
        # self.points_logged = False
        
        # self.cluster_distance_threshold = 0.4
        # self.cluster_min_points = 2

        self.tf_buffer = Buffer() # this is to store the transforms that we will get from the TransformListener
        
        self.declare_parameter('target_frame', 'base_footprint')

        self.declare_parameter('roi_x_min', 0.0)
        self.declare_parameter('roi_x_max', 25.0)

        self.declare_parameter('roi_y_min', -8.0)
        self.declare_parameter('roi_y_max', 8.0)

        self.declare_parameter('roi_z_min', -0.2)
        self.declare_parameter('roi_z_max', 1.5)

        self.declare_parameter('ground_threshold', 0.01)

        self.declare_parameter('cluster_distance_threshold', 0.30)
        self.declare_parameter('cluster_min_points', 2)
        
        self.target_frame = self.get_parameter('target_frame').value

        self.roi_x_min = self.get_parameter('roi_x_min').value
        self.roi_x_max = self.get_parameter('roi_x_max').value

        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_y_max = self.get_parameter('roi_y_max').value

        self.roi_z_min = self.get_parameter('roi_z_min').value
        self.roi_z_max = self.get_parameter('roi_z_max').value

        self.ground_threshold = self.get_parameter(
            'ground_threshold'
        ).value

        self.cluster_distance_threshold = self.get_parameter(
            'cluster_distance_threshold'
        ).value

        self.cluster_min_points = self.get_parameter(
            'cluster_min_points'
        ).value
        
        
        self.subscription = self.create_subscription(
            PointCloud2,
            '/velodyne_points',
            self.lidar_callback,
            10
        )

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self
        ) 
        self.roi_publisher = self.create_publisher(
            PointCloud2,
            '/perception/roi_points',
            10
        )
        self.non_ground_publisher = self.create_publisher(
            PointCloud2,
            '/perception/non_ground_points',
            10
        )
        self.clustered_publisher = self.create_publisher(
            PointCloud2,
            '/perception/clustered_points',
            10
        )

        self.get_logger().info('LiDAR node started')
        
    def lidar_callback(self, msg):

        source_frame = msg.header.frame_id

        if not self.frame_logged:
            self.get_logger().info(
                f'LiDAR source frame: {source_frame}'
            )
            self.frame_logged = True

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                rclpy.time.Time() #get the latest transform available
            )

            if not self.transform_logged:
                self.get_logger().info(
                    f'TF found: {source_frame} -> {self.target_frame}'
                )
                self.transform_logged = True

        except TransformException as ex:
            self.get_logger().warning(
                f'Could not transform {source_frame} '
                f'to {self.target_frame}: {ex}'
            )
            return
        points = point_cloud2.read_points_numpy(
            msg,
            field_names=('x', 'y', 'z', 'intensity'),
            skip_nans=True
        )

        xyz = points[:, :3]
        intensity = points[:, 3]
        xyz_base = transform_points(
            xyz,
            transform
        )
        # if not self.points_logged:
        #     self.get_logger().info(
        #         f'First transformed points:\n{xyz_base[:5]}'
        #     )
        #     min_xyz = np.min(xyz_base, axis=0)
        #     max_xyz = np.max(xyz_base, axis=0)

        #     self.get_logger().info(
        #         f'Point cloud range in base_footprint:\n'
        #         f'X: {min_xyz[0]:.2f} to {max_xyz[0]:.2f} m\n'
        #         f'Y: {min_xyz[1]:.2f} to {max_xyz[1]:.2f} m\n'
        #         f'Z: {min_xyz[2]:.2f} to {max_xyz[2]:.2f} m'
        #     )
        #     self.points_logged = True   
        roi_xyz, roi_intensity = extract_roi(
            xyz_base,
            intensity,
            self.roi_x_min,
            self.roi_x_max,
            self.roi_y_min,
            self.roi_y_max,
            self.roi_z_min,
            self.roi_z_max
        )
        
        # if not self.points_logged:
        #     self.get_logger().info(
        #         f'Points before ROI: {len(xyz_base)}'
        #     )

        #     self.get_logger().info(
        #         f'Points after ROI: {len(roi_xyz)}'
        #     )

        roi_points = np.column_stack((
            roi_xyz,
            roi_intensity
        ))
        fields = [
            PointField(
                name='x',
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='y',
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='z',
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name='intensity',
                offset=12,
                datatype=PointField.FLOAT32,
                count=1
            )
        ]# the same fields as the original PointCloud2 message
        
        roi_header = Header()

        roi_header.stamp = msg.header.stamp
        roi_header.frame_id = self.target_frame
        roi_msg = point_cloud2.create_cloud(
            roi_header,
            fields,
            roi_points
        )

        self.roi_publisher.publish(roi_msg)        
        # for threshold in [0.02, 0.04, 0.06, 0.08, 0.10]:
        #     count = np.sum(roi_xyz[:, 2] > threshold)

        #     self.get_logger().info(
        #         f'Points above {threshold:.2f} m: {count}'
        #     )
        non_ground_xyz, non_ground_intensity = remove_ground(
            roi_xyz,
            roi_intensity,
            ground_threshold=self.ground_threshold
        )
        
        # self.get_logger().info(
        #     f'Points after ground removal: {len(non_ground_xyz)}'
        # )
        non_ground_points = np.column_stack((
            non_ground_xyz,
            non_ground_intensity
        ))
        
        non_ground_msg = point_cloud2.create_cloud(
            roi_header,
            fields,
            non_ground_points
        )
        
        self.non_ground_publisher.publish(non_ground_msg)     
        
        cluster_labels = euclidean_clustering(
            non_ground_xyz,
            self.cluster_distance_threshold,
            self.cluster_min_points
        )
        
        # valid_labels = cluster_labels[cluster_labels >= 0]

        # if len(valid_labels) > 0:
        #     number_of_clusters = len(np.unique(valid_labels))
        # else:
        #     number_of_clusters = 0

        # self.get_logger().info(
        #     f'Clusters found: {number_of_clusters}'
        # )
        valid_cluster_mask = cluster_labels >= 0

        clustered_xyz = non_ground_xyz[valid_cluster_mask]

        clustered_labels = cluster_labels[valid_cluster_mask]
        
        clustered_points = np.column_stack((
            clustered_xyz,
            clustered_labels.astype(np.float32)
        ))
        
        clustered_msg = point_cloud2.create_cloud(
            roi_header,
            fields,
            clustered_points
        )

        self.clustered_publisher.publish(clustered_msg)


def main(args=None):
    rclpy.init(args=args)

    node = LidarNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()