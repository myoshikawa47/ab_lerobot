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
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        
class Deploy(RTCore):
    def __init__(self, args: argparse.Namespace):
        super(Deploy, self).__init__()
        freq = args.freq
        self.r = rospy.Rate(freq)
            
        # load LeRobotDataset
        dataset = LeRobotDataset('null', root=args.data_dir)
        start = dataset.episode_data_index['from'][args.idx].item()
        end = dataset.episode_data_index['to'][args.idx].item()
        frames = [dataset[i] for i in range(start, end)]
        # self.nloop = len(frames)
        self.nloop = args.exp_time * freq
        self.gt = {k: np.array([frame[k].cpu() for frame in frames]) for k, v in frames[0].items() if isinstance(v, torch.Tensor)}
        _init_pos = self.gt['observation.state'][0]
        self.init_pos = {'head': _init_pos[:2], 'left_arm': _init_pos[2:11], 'right_arm': _init_pos[11:20]}
        # self.init_pos = {'left_arm': _init_pos[2:11], 'right_arm': _init_pos[11:20]}
        # gt['observation.state.pos'] = np.concatenate([gt['observation.state.head'], gt['observation.state.left_arm.pos'], gt['observation.state.right_arm.pos']], axis=-1)
        # gt['observation.state.torque'] = np.concatenate([gt['observation.state.left_arm.torque'], gt['observation.state.right_arm.torque']], axis=-1)
        # gt['observation.state.tactile'] = np.concatenate([gt['observation.state.left_arm.tactile'], gt['observation.state.right_arm.tactile']], axis=-1)
        # gt['action'] = np.concatenate([gt['action.head'], gt['action.left_arm'], gt['action.right_arm']], axis=-1)
        # set device 

        device = 'cpu' if args.device < 0 else f"cuda:{args.device}"

        # load policy            
        policy: PreTrainedPolicy = Policy.from_pretrained(args.ckpt_folder)
        # policy: PreTrainedPolicy = SmolVLAPolicy.from_pretrained(args.ckpt_folder)
        # policy: PreTrainedPolicy = ACTPolicy.from_pretrained(args.ckpt_folder)
        # policy: PreTrainedPolicy = DiffusionPolicy.from_pretrained(args.ckpt_folder)
        policy.to(device)
        policy.eval()
        policy.reset()
        
        # We can verify that the shapes of the features expected by the policy match the ones from the observations
        # produced by the environment
        print(policy.config.input_features)

        # Similarly, we can check that the actions produced by the policy will match the actions expected by the
        # environment
        print(policy.config.output_features)
        
        self.args = args
        self.device = device
        self.policy = policy
        # self.task = 'Put the ladle in the sink.'
        # self.task = "Pick up the black ladle and place it in the sink."
        # self.task = "Pick up the frying pan and place it in the sink."
        # self.task = "Pick up the green ladle and place it in the sink."
        # self.task = "Pick up the red ladle and place it in the sink."
        self.task = "Pick up the steel bowl and place it in the sink."
        # self.task = "Pick up the white bowl and place it in the sink."
        # self.task = "Pick up the white plate and place it in the sink."
        self.observation = {
            # 'observation.image.head.left': None,
            'observation.image.head.right': None,
            'observation.image.arm.left': None,
            'observation.image.arm.right': None,
            'observation.state.head': None,
            'observation.state.left_arm.pos': None,
            'observation.state.right_arm.pos': None,
            # 'observation.state': None,
            # 'task': None
            }
        self.obs_name_to_msg_name = {
            'observation.image.head.left': 'img_head_left',
            'observation.image.head.right': 'img_head_right',
            'observation.image.arm.left': 'img_arm_left',
            'observation.image.arm.right': 'img_arm_right',
            'observation.state.head': 'head_state',
            'observation.state.left_arm.pos': 'left_arm_state',
            'observation.state.right_arm.pos': 'right_arm_state',
            # 'task': 'task',
        }
 
    def initialization(self, exp_time: int = 5, freq: int = 100,) -> None:
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
            
    def pub_msg(self, target_pos: dict[str, np.ndarray]) -> None:
        for k, v in target_pos.items():
            msg = getattr(self, k + '_msg')
            pub = getattr(self, k + '_pub')
            msg.header.stamp = rospy.Time.now()
            msg.position = v
            pub.publish(msg)
            
    def get_observation(self, loop_ct: int | None = None) -> dict[str, torch.Tensor]:
        for k in self.observation.keys():
            self.observation[k] = getattr(self, self.obs_name_to_msg_name[k])
        if loop_ct is not None:
            self.observation = {k: v[loop_ct] for k, v in self.gt.items() if isinstance(v, np.ndarray)}
        obs = dict()
        obs = {k: torch.Tensor(v).unsqueeze(0).to(self.device) for k, v in self.observation.items() if isinstance(v, np.ndarray)}
        obs['observation.state'] = torch.Tensor(np.concatenate([self.observation['observation.state.head'],
                                                  self.observation['observation.state.left_arm.pos'],
                                                  self.observation['observation.state.right_arm.pos']], axis=-1)).unsqueeze(0).to(self.device)
        obs['task'] = self.task
        return obs
    
    def run(self):
        input("Playback.run(): Press enter to move to init pos")
        self.initialization()
        
        input("Are you ready?")
        pred_count = 0
        for loop_ct in range(self.nloop):
            # _target_pos = self.gt['observation.state'][loop_ct]
            obs = self.get_observation()
            if self.args.policy_type == 'smolvla' and len(self.policy._queues['action']) == 0:
                pred_count += 1
                print(f'prediction: {pred_count}')
            action = self.policy.select_action(obs).squeeze(0).cpu().numpy()
            target_pos = {'head': action[:2], 'left_arm': action[2:11], 'right_arm': action[11:20]}
            # print(target_pos)
            self.pub_msg(target_pos)
            self.r.sleep()
            
        rospy.logwarn("Playback.run(): Finished execution")

        self.initialization()
        
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_folder", type=str, default=None)
    parser.add_argument("--policy_type", choices=['smolvla', 'act', 'diffusion'], default='smolvla')
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--exp_time", type=int, default=30)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--device", type=int, default="0")
    args = parser.parse_args()
    
    if args.policy_type == 'smolvla':
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
    elif args.policy_type == 'act':
        from lerobot.policies.act.modeling_act import ACTPolicy as Policy
    elif args.policy_type == 'diffusion':
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Policy
    
    try:
        rospy.init_node("playback_node", anonymous=True)
        task = Deploy(args)
        time.sleep(1)
        task.run()
        sys.exit()
    except rospy.ROSInterruptException or KeyboardInterrupt or EOFError as e:
        init_exptime=3
        init_freq=100
        nloop = init_exptime * init_freq
        
        current_position = np.array(task.upper_msg.position)
        target_position = np.array(task.upper_msg.position)
        target_position[11] = 223
        
        trajectory = np.linspace(current_position, target_position , nloop)
        for i in range(nloop):        
            task.upper_msg.header.stamp = rospy.Time.now()
            task.upper_msg.position = trajectory[i]
            # task.upperbody_pub.publish(task.upper_msg)
            time.sleep(1./init_freq)
