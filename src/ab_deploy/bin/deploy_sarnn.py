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

from eipl.model import SARNN
from eipl.utils import restore_args, tensor2numpy, deprocess_img, resize_img, normalization

        
class Deploy(RTCore):
    def __init__(self, args: argparse.Namespace, params: dict):
        super(Deploy, self).__init__()
        freq = args.freq
        self.r = rospy.Rate(freq)
            
        # load LeRobotDataset
        dataset = LeRobotDataset('null', root=args.data_dir)
        start = dataset.episode_data_index['from'][args.idx].item()
        end = dataset.episode_data_index['to'][args.idx].item()
        frames = [dataset[i] for i in range(start, end)]
        self.nloop = len(frames)
        self.nloop = args.exp_time * args.freq
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

        # define model
        model = SARNN(
            rec_dim=params["rec_dim"],
            joint_dim=20,
            k_dim=params["k_dim"],
            heatmap_size=params["heatmap_size"],
            temperature=params["temperature"],
            im_size=[36, 64],
        )
        if params["compile"]:
            model = torch.compile(model)

        # load weight
        ckpt = torch.load(args.filename, map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        model.eval()        
        model.to(device)

        self.device = device
        # self.state_bounds = (model.meta.stats[params["observation.state"]]["min"], model.meta.stats[params["observation.state"]]["max"])
        self.state_bounds = (dataset.meta.stats["observation.state"]["min"], dataset.meta.stats["observation.state"]["max"])
        self.action_bounds = (dataset.meta.stats["action"]["min"], dataset.meta.stats["action"]["max"])
        self.minmax = (params['vmin'], params['vmax'])
        self.model = model
        self.task = 'Put the ladle in the sink.'
        # self.task = 'Pick up the ladle on the table and put it in the sink.'
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
        state = None
        with torch.inference_mode():
            for loop_ct in range(self.nloop):
                # _target_pos = self.gt['observation.state'][loop_ct]
                obs = self.get_observation()
                nimg_t = obs[params['img_key']]
                joint_t = obs[params['state_key']]
                # nimg_t = torch.from_numpy(cv2.resize(np.array(obs[params['img_key']][0]), (64, 36))).unsqueeze(0)
                # joint_t = torch.from_numpy(cv2.resize(np.array(obs[params['state_key'][0]]), (64, 36))).unsqueeze(0)
                
                # # closed loop
                # if loop_ct > 0:
                #     nimg_t = args.input_param * nimg_t + (1.0 - args.input_param) * ny_image
                #     joint_t = args.input_param * joint_t + (1.0 - args.input_param) * action

                #normalization
                njoint_t = normalization(joint_t, self.state_bounds, self.minmax).to(torch.float32)
                ny_image, naction, enc_pts, dec_pts, state = self.model(nimg_t, njoint_t, state)
                action = normalization(naction[0].cpu().numpy(), self.minmax, self.action_bounds)
                target_pos = {'head': action[:2], 'left_arm': action[2:11], 'right_arm': action[11:20]}
                print(target_pos)
                self.pub_msg(target_pos)
                self.r.sleep()
            
        rospy.logwarn("Playback.run(): Finished execution")

        self.initialization()
        
        
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--input_param", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--exp_time", type=int, default=30)
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    
    # restore parameters
    dir_name = os.path.split(args.filename)[0]
    params = restore_args(os.path.join(dir_name, "args.json"))
    
    try:
        rospy.init_node("deploy_node", anonymous=True)
        task = Deploy(args, params)
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
