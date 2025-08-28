#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import argparse
import time
import numpy as np
import rospy
import torch

import ipdb
import glob
# from natsort import natsorted
# from cv_bridge import CvBridge

# from eipl.model import SARNN
# from eipl.utils import normalization
# from eipl.utils import restore_args, tensor2numpy, deprocess_img

# ROS
import rospy
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage


class RTCore:
    def __init__(self):
        # load initial data
        # self.img_head_left = None
        self.img_head_right = None
        self.img_arm_left = None
        self.img_arm_right = None
        self.head_state = None
        self.left_arm_state = None
        self.right_arm_state = None
        self.task_index = None # to be overwritten
        # self.bridge = CvBridge()
        
        self.head_msg = JointState()
        self.left_arm_msg = JointState()
        self.right_arm_msg = JointState()
        
        self.head_pub = rospy.Publisher("/maharo/head/command_states", JointState, queue_size=1)
        self.left_arm_pub = rospy.Publisher("/maharo/left_arm/upperbody/command_states", JointState, queue_size=1)
        self.right_arm_pub = rospy.Publisher("/maharo/right_arm/upperbody/command_states", JointState, queue_size=1)

        rospy.Subscriber("/maharo/head/joint_states", JointState, self.head_callback)
        rospy.Subscriber("/maharo/left_arm/upperbody/joint_states", JointState, self.left_arm_state_callback)
        rospy.Subscriber("/maharo/right_arm/upperbody/joint_states", JointState, self.right_arm_state_callback)
        # rospy.Subscriber("/zed/zed_node/left/image_rect_color", Image, self.left_img_callback)
        # rospy.Subscriber("/zed/zed_node/right/image_rect_color", Image, self.right_img_callback)
        rospy.Subscriber("/zed2i/zed_node/stereo_raw/image_raw_color/compressed", CompressedImage, self.stereo_img_callback)
        rospy.Subscriber("/left_hand_camera/image_raw/compressed", CompressedImage, self.left_hand_img_callback)
        rospy.Subscriber("/right_hand_camera/image_raw/compressed", CompressedImage, self.right_hand_img_callback)

        print("Waiting for message...")
        rospy.wait_for_message('/zed2i/zed_node/stereo_raw/image_raw_color/compressed', CompressedImage, timeout=None)
        rospy.wait_for_message('/left_hand_camera/image_raw/compressed', CompressedImage, timeout=None)
        rospy.wait_for_message('/right_hand_camera/image_raw/compressed', CompressedImage, timeout=None)
        # time.sleep(1)
    
    def stereo_img_callback(self, msg):
        img_arr = np.frombuffer(msg.data, np.uint8)
        np_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        # self.img_head_left = np.transpose(np_img[:, :int(np_img.shape[1]/2)] / 255, (2,0,1))
        # self.img_head_right = np.transpose(np_img[:, int(np_img.shape[1]/2):] / 255, (2,0,1))
        self.img_head_right = np.transpose(cv2.resize(np_img[:, int(np_img.shape[1]/2):], (64, 36)) / 255, (2,0,1)).astype(np.float32)
        # ######### DEBUG ##########
        # self.img_head_right = np.transpose(np_img[:, :int(np_img.shape[1]/2)] / 255, (2,0,1))
        # ##########################

    def left_hand_img_callback(self, msg):
        img_arr = np.frombuffer(msg.data, np.uint8)
        np_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)                        
        self.img_arm_left = np.transpose(cv2.rotate(np_img[:360], cv2.ROTATE_180) / 255, (2,0,1))
        
    def right_hand_img_callback(self, msg):
        img_arr = np.frombuffer(msg.data, np.uint8)
        np_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)                        
        self.img_arm_right = np.transpose(cv2.rotate(np_img[:360], cv2.ROTATE_180) / 255, (2,0,1))
        
    def head_callback(self, msg):
        if len(self.head_msg.name) == 0:
            self.head_msg.name = msg.name
            self.head_msg.position = msg.position
        self.head_state = np.array(msg.position)
        
    def left_arm_state_callback(self, msg):
        if len(self.left_arm_msg.name) == 0:
            self.left_arm_msg.name = msg.name
            self.left_arm_msg.position = msg.position
        self.left_arm_state = np.array(msg.position)
    
    def right_arm_state_callback(self, msg):
        if len(self.right_arm_msg.name) == 0:
            self.right_arm_msg.name = msg.name
            self.right_arm_msg.position = msg.position
        self.right_arm_state = np.array(msg.position)
        
    def custom_bridge(self, msg):
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        height, width = msg.height, msg.width
        
        # 画像のデコード（msg.encoding に応じて変換方法を変更する）
        if msg.encoding == "rgb8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV は BGR 形式
            return img
        elif msg.encoding == "bgr8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return img
        elif msg.encoding == "mono8":
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            return img
        elif msg.encoding == "bgra8":
            img = np_arr.reshape((height, width, 4))  # BGRA形式
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # αチャンネルを削除してBGRに変換
            return img
        else:
            rospy.logerr(f"Unsupported encoding: {msg.encoding}")
            return
