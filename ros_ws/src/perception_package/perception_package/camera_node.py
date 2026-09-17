import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
)
"""
Node: Camera Perception

Input:
    /zed/left/image_rect_color

Output:
    /perception/yolo_bboxes
"""

class CameraNode(Node):

    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()
        package_share = get_package_share_directory('perception_package')

        model_path = Path(package_share) / 'models' / 'yolov26.pt'

        self.model = YOLO(str(model_path))
        self.subscription = self.create_subscription(
            Image,
            '/zed/left/image_rect_color',
            self.image_callback,
            10
        )
        self.detections_publisher = self.create_publisher(
            Detection2DArray,
            '/perception/yolo_bboxes',
            10
        )

        self.get_logger().info('Camera node started')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='bgr8'
        )
        detections_msg = Detection2DArray()
        detections_msg.header = msg.header

        results = self.model(cv_image, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detection = Detection2D()
                detection.header = msg.header

                detection.bbox.center.position.x = (x1 + x2) / 2.0
                detection.bbox.center.position.y = (y1 + y2) / 2.0
                detection.bbox.size_x = x2 - x1
                detection.bbox.size_y = y2 - y1

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence

                detection.results.append(hypothesis)

                detections_msg.detections.append(detection)

                # self.get_logger().info(
                #     f'Detected: {class_name}, '
                #     f'confidence: {confidence:.2f}, '
                #     f'box: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})'
                # )
        self.detections_publisher.publish(detections_msg)
def main(args=None):
    rclpy.init(args=args)

    node = CameraNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()