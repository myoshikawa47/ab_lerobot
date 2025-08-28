#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from IPython.utils.path import target_outdated
import cv2

import sys
import argparse
import time
import numpy as np
import rospy
import torch

import ipdb
import glob
import json
from datetime import datetime

# ROS
import rospy
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header

from rt_core import RTCore

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

        
class Playback(RTCore):
    def __init__(self, args):
        super(Playback, self).__init__()
        freq = args.freq
        self.r = rospy.Rate(freq)
            
        # load LeRobotDataset
        dataset = LeRobotDataset('null', root=args.data_dir)
        start = dataset.episode_data_index['from'][args.idx].item()
        end = dataset.episode_data_index['to'][args.idx].item()
        frames = [dataset[i] for i in range(start, end)]
        self.nloop = len(frames)
        self.gt = {k: np.array([frame[k].cpu() for frame in frames]) for k, v in frames[0].items() if isinstance(v, torch.Tensor)}
        _init_pos = self.gt['observation.state'][0]
        self.init_pos = {'head': _init_pos[:2], 'left_arm': _init_pos[2:11], 'right_arm': _init_pos[11:20]}
        # self.init_pos = {'left_arm': _init_pos[2:11], 'right_arm': _init_pos[11:20]}
        # import ipdb;ipdb.set_trace()
        # gt['observation.state.pos'] = np.concatenate([gt['observation.state.head'], gt['observation.state.left_arm.pos'], gt['observation.state.right_arm.pos']], axis=-1)
        # gt['observation.state.torque'] = np.concatenate([gt['observation.state.left_arm.torque'], gt['observation.state.right_arm.torque']], axis=-1)
        # gt['observation.state.tactile'] = np.concatenate([gt['observation.state.left_arm.tactile'], gt['observation.state.right_arm.tactile']], axis=-1)
        # gt['action'] = np.concatenate([gt['action.head'], gt['action.left_arm'], gt['action.right_arm']], axis=-1)
    
    def initialization(self, exp_time=10, freq=100,):
        rospy.logwarn("Start initialization")
        nloop = exp_time * freq

        traj = dict()
        for k, v in self.init_pos.items():
            curr_pos = getattr(self, k + '_msg').position
            traj[k] = np.linspace(curr_pos, v, nloop)
        for i in range(1, nloop):
            target_pos = {k: v[i] for k, v in traj.items()}
            self.pub_msg(target_pos)
            time.sleep(1./freq)
        
        rospy.logwarn("Finished initialization")
            
    def pub_msg(self, target_pos):
        for k, v in target_pos.items():
            msg = getattr(self, k + '_msg')
            pub = getattr(self, k + '_pub')
            msg.header.stamp = rospy.Time.now()
            msg.position = v
            pub.publish(msg)
    
    def run(self):
        input("Playback.run(): Press enter to move to init pos")
        self.initialization()
        
        input("Are you ready?")

        for loop_ct in range(self.nloop):
            _target_pos = self.gt['observation.state'][loop_ct]
            target_pos = {'head': _target_pos[:2], 'left_arm': _target_pos[2:11], 'right_arm': _target_pos[11:20]}
            # target_pos = {'left_arm': _target_pos[2:11], 'right_arm': _target_pos[11:20]}
            print(target_pos)
            self.pub_msg(target_pos)
            self.r.sleep()
            
        rospy.logwarn("Playback.run(): Finished execution")

        self.initialization()        
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--freq", type=int, default=10)
    args = parser.parse_args()
    
    rospy.init_node("playback_node", anonymous=True)
    task = Playback(args)
    time.sleep(1)
    task.run()
    sys.exit()