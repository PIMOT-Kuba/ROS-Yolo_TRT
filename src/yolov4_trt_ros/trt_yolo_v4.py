#!/usr/bin/env python3

import os
from time import time

import numpy as np
import cv2
import pycuda.autoinit  # For initializing CUDA driver
import pycuda.driver as cuda

from lidar_data_writer import LidarDataWriter
from utils.yolo_classes import get_cls_dict
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

import rospy
import rospkg
import message_filters
from yolov4_trt_ros.msg import Detector2DArray
from yolov4_trt_ros.msg import Detector2D
from vision_msgs.msg import BoundingBox2D
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2


class yolov4(object):
    def __init__(self):
        """ Constructor """

        # self.bridge = CvBridge()
        self.init_params()
        self.init_yolo()
        self.cuda_ctx = cuda.Device(0).make_context()
        self.trt_yolo = TrtYOLO(
            (self.model_path + self.model), (self.h, self.w), self.category_num)

    def __del__(self):
        """ Destructor """
        
        self.cuda_ctx.pop()
        del self.trt_yolo
        del self.cuda_ctx

    def clean_up(self):
        """ Backup destructor: Release cuda memory """

        if self.trt_yolo is not None:
            self.cuda_ctx.pop()
            del self.trt_yolo
            del self.cuda_ctx

    def init_params(self):
        """ Initializes ros parameters """
        
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("yolov4_trt_ros")

        self.video_topic = rospy.get_param("/video_topic", "/zed/zed_node/rgb_raw/image_raw_color")
        self.lidar_topic = rospy.get_param("/lidar_topic", "/velodyne_points")
        self.model = rospy.get_param("/model", "yolov4-416")
        self.model_path = rospy.get_param("/model_path", package_path + "/yolo/")
        self.category_num = rospy.get_param("/category_number", 5)
        self.input_shape = rospy.get_param("/input_shape", "416")
        self.conf_th = rospy.get_param("/confidence_threshold", 0.5)
        self.show_img = rospy.get_param("/show_image", True)

        self.image_sub = message_filters.Subscriber(self.video_topic, Image, queue_size=1, buff_size=1920*1080*3)
        self.lidar_sub = message_filters.Subscriber('/velodyne_points', PointCloud2, queue_size=1)
        print('Starting...')

        # slop - delay in seconds with which the messages can be synchronized
        synchronized_sub = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.lidar_sub], queue_size=1,
                                                                slop=0.001, allow_headerless=True)
        # Callback function
        synchronized_sub.registerCallback(callback  )

        # Logging and saving
        self.log_time = rospy.get_param("/log_time", True)
        self.save_frames = rospy.get_param("/save_frames", True)
        self.save_lidar = rospy.get_param("/save_lidar", True)
        self.log_dir = 'log/'
        self.frame_counter = 0
        
        if not os.path.exists(self.log_dir()):
            os.makedirs(self.log_dir)

        if self.log_time:
            self.log_file_path = os.path.join(self.log_dir, 'detection_times.log')

        if self.save_frames:
            self.log_camera_dir = os.path.join(self.log_dir)

        if self.save_lidar:
            self.log_lidar_dir = os.path.join(self.log_dir, 'lidar')
            if not os.path.exists(self.log_lidar_dir):
                os.makedirs(self.log_lidar_dir)
            self.lidar_data_writer = LidarDataWriter(self.log_lidar_dir)

        # Publishers
        self.detection_pub = rospy.Publisher(
            "detections", Detector2DArray, queue_size=1)
        self.overlay_pub = rospy.Publisher(
            "/result/overlay", Image, queue_size=1)


    def init_yolo(self):
        """ Initialises yolo parameters required for trt engine """

        if self.model.find('-') == -1:
            self.model = self.model + "-" + self.input_shape
            
        yolo_dim = self.model.split('-')[-1]

        if 'x' in yolo_dim:
            dim_split = yolo_dim.split('x')
            if len(dim_split) != 2:
                raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
            self.w, self.h = int(dim_split[0]), int(dim_split[1])
        else:
            self.h = self.w = int(yolo_dim)
        if self.h % 32 != 0 or self.w % 32 != 0:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

        cls_dict = get_cls_dict(self.category_num)
        self.vis = BBoxVisualization(cls_dict)


    def callback(self, ros_img, lidar_data):
        """Continuously capture images from camera and do object detection """

        # converts from ros_img to cv_img for processing
        cv_img = np.frombuffer(ros_img.data, dtype=np.uint8).reshape(ros_img.height, ros_img.width, -1)

        if cv_img is not None:
            img_preprocessed = self.trt_yolo.preprocess_yolo(cv_img, self.trt_yolo.input_shape)
            
            # MEASURE DETECTION TIME
            start_time = time()
            boxes, confs, clss = self.trt_yolo.detect(img_preprocessed, (cv_img.shape[0], cv_img.shape[1]), self.conf_th)
            detection_time = round((time() - start_time) * 1000, 5)

            cv_img = self.vis.draw_bboxes(cv_img, boxes, confs, clss)
            
            fps = 1.0 / detection_time

            if self.log_time:
                self.frame_counter += 1
                text_info = f'FRAME: {self.frame_counter}, {detection_time} ms\n'
                print(text_info)

                self.file_handler = open(self.log_file_path, 'a')
                self.file_handler.write(text_info)
                self.file_handler.close()

            self.publisher(boxes, confs, clss)

            if self.save_frames:
                img_filename = f'frame_{self.frame_counter}.jpg'
                cv2.imwrite(os.path.join(self.log_camera_dir, img_filename))

            if self.save_lidar:
                preprocessed_lidar_data = self.lidar_data_writer.preprocess(lidar_data)
                lidar_filename = f'lidar_{self.frame_counter}.bin'
                self.lidar_data_writer.save_data(preprocessed_lidar_data, os.path.join(self.log_lidar_dir, lidar_filename))

            if self.show_img:
                cv_img = show_fps(cv_img, fps)
                cv2.imshow("YOLOv4 DETECTION RESULTS", cv_img)
                cv2.waitKey(1)

    def publisher(self, boxes, confs, clss):
        """ Publishes to detector_msgs

        Parameters:
        boxes (List(List(int))) : Bounding boxes of all objects
        confs (List(double))	: Probability scores of all objects
        clss  (List(int))	: Class ID of all classes
        """
        detection2d = Detector2DArray()
        detection = Detector2D()
        detection2d.header.stamp = rospy.Time.now()
        detection2d.header.frame_id = "camera" # change accordingly
        
        for i in range(len(boxes)):
            # boxes : xmin, ymin, xmax, ymax
            for _ in boxes:
                detection.header.stamp = rospy.Time.now()
                detection.header.frame_id = "camera" # change accordingly
                detection.results.id = clss[i]
                detection.results.score = confs[i]

                detection.bbox.center.x = boxes[i][0] + (boxes[i][2] - boxes[i][0])/2
                detection.bbox.center.y = boxes[i][1] + (boxes[i][3] - boxes[i][1])/2
                detection.bbox.center.theta = 0.0  # change if required

                detection.bbox.size_x = abs(boxes[i][0] - boxes[i][2])
                detection.bbox.size_y = abs(boxes[i][1] - boxes[i][3])

            detection2d.detections.append(detection)
        
        self.detection_pub.publish(detection2d)


def main():
    yolo = yolov4()
    rospy.init_node('yolov4_trt_ros', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.on_shutdown(yolo.clean_up())
        print("Shutting down")


if __name__ == '__main__':
    main()
