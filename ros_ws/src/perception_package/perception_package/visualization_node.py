import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_interfaces.msg import BoundingBoxArray

import cv2


from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import math

"""
Node: Visualization

Inputs:
    /zed/left/image_rect_color
    /perception/yolo_bboxes
    /ground_truth/odom
    /imu/data

Output:
    /perception/debug_image
"""

class VisualizationNode(Node):

    def __init__(self):
        super().__init__('visualization_node')
        self.latest_odom = None
        self.latest_imu = None

        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image,
            '/zed/left/image_rect_color',
            self.image_callback,
            10
        )

        self.detections_subscription = self.create_subscription(
            BoundingBoxArray,
            '/perception/yolo_bboxes',
            self.detections_callback,
            10
        )
        
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/ground_truth/odom',
            self.odom_callback,
            10
        )

        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        self.debug_image_publisher = self.create_publisher(
            Image,
            '/perception/debug_image',
            10
        )

        self.image_cache = {}
        self.get_logger().info('Visualization node started')
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )

        timestamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

        self.image_cache[timestamp] = cv_image ## so we can match the image with the detections later from the same timestamp
        
    def detections_callback(self, msg):

        timestamp = ( msg.header.stamp.sec, msg.header.stamp.nanosec)

        # Find the camera image that has the same timestamp
        if timestamp not in self.image_cache:
            self.get_logger().warning(
                f'No matching image for timestamp {timestamp}'
            )
            return

        cv_image = self.image_cache.pop(timestamp)

        class_info = {
            0: ('blue_cone', (255, 0, 0)),
            1: ('yellow_cone', (0, 255, 255)),
            2: ('orange_cone', (0, 165, 255)),
            3: ('large_orange_cone', (0, 100, 255)),
            4: ('unknown_cone', (255, 255, 255))
        }

        for box in msg.boxes:

            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)

            class_name, color = class_info.get(
                box.class_id,
                ('unknown', (255, 255, 255)) # fallback value for unknown classes
            )

            # Draw bounding rectangle
            cv2.rectangle(
                cv_image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f'{class_name} {box.confidence:.2f}'

            # Keep the text inside the image
            text_y = max(y1 - 8, 20)

            cv2.putText(
                cv_image,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
        if self.latest_odom is not None:

            position = self.latest_odom.pose.pose.position
            velocity = self.latest_odom.twist.twist.linear

            speed = math.sqrt(
                velocity.x ** 2 +
                velocity.y ** 2 +
                velocity.z ** 2
            )

            position_text = (
                f'Position: x={position.x:.2f}, '
                f'y={position.y:.2f}, '
                f'z={position.z:.2f}'
            )

            speed_text = f'Speed: {speed:.2f} m/s'

            cv2.putText(
                cv_image,
                position_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                cv_image,
                speed_text,
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
        if self.latest_imu is not None:

            q = self.latest_imu.orientation

            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

            yaw = math.atan2(siny_cosp, cosy_cosp)
            yaw_degrees = math.degrees(yaw)

            orientation_text = f'Yaw: {yaw_degrees:.1f} deg'

            cv2.putText(
                cv_image,
                orientation_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        debug_msg = self.bridge.cv2_to_imgmsg(
            cv_image,
            encoding='bgr8'
        )

        debug_msg.header = msg.header

        self.debug_image_publisher.publish(debug_msg)
    
    def odom_callback(self, msg):
        self.latest_odom = msg


    def imu_callback(self, msg):
        self.latest_imu = msg

def main(args=None):
    rclpy.init(args=args)

    node = VisualizationNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()