#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from IPython.utils.path import target_outdated
import cv2
import matplotlib
import matplotlib.pyplot as plt

import ipdb.stdout
matplotlib.use('Agg')
import sys
import argparse
import time
import numpy as np
import rospy
import torch

import ipdb
import glob
from natsort import natsorted
from scipy.interpolate import CubicSpline
import json
from datetime import datetime
# from PIL import Image as pil_image


from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img

# ROS
import rospy
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header

from rt_core import RTCore

# local
sys.path.append("../")
from util import Visualize, Processor, Deprocessor, RTSelector
        
class RTControl(RTCore):
    def __init__(self, args):
        super(RTControl, self).__init__()
        freq = args.freq
        exptime = args.exptime
        
        self.r = rospy.Rate(freq)
        self.nloop = freq * exptime
        self.open_ratio = args.open_ratio
        
        # set device 
        try:
            if args.device >= 0:
                device = "cuda:{}".format(args.device)
            else:
                device = "cpu"
        except TypeError as e:
            device = "cpu"
        
        # restore parameters
        # dir_name = os.path.split(filename)[0]
        log_dir_path = f"../log/{args.log_dir_name}"
        # params = restore_args(os.path.join(dir_name, "args.json"))
        try:
            params = restore_args(os.path.join(log_dir_path, "args.json"))
        except FileNotFoundError as e:
            print("no such file!")
            exit()
        
        minmax = [params["data"]["vmin"], params["data"]["vmax"]]
        stdev = params["data"]["stdev"] * (params["data"]["vmax"] - params["data"]["vmin"])
        
        img_bounds = [0.0,255.0]
        # self.vec_bounds = np.load(f"../data/20250422/param/arm_state_bound.npy")
        # self.vec_bounds = np.array([[2022., 1436., 2029.,  942., 2119., 2024., 1637., 2002.,  230.],
        #                                     [3113., 2089., 3062., 1973., 3367., 2181., 2200., 2106., 2326.]],
        #                                     dtype=np.float32)
        self.vec_bounds = [304.0,3235.0]
        
        self.eye_img_size = params["data"]["img_size"]
        self.processor = Processor(img_bounds, self.vec_bounds, minmax)
        model_name = params["model"]["model_name"]

        selector = RTSelector(params, device)
        self.model = selector.select_model()

        weight_pathes = natsorted(glob.glob(f"{log_dir_path}/*.pth"))
        
        ckpt = torch.load(weight_pathes[args.ckpt_idx], map_location=torch.device("cpu"), weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        
        self.deprocessor = Deprocessor(img_bounds, self.vec_bounds, minmax, select_idxs=[0,])
        self.key_dim = params["model"]["key_dim"]
        
        # self.grasp_open_val = 223
        # self.grasp_close_val = 2173
        # # self.grasp_thresh_ratio = args.grasp_thresh_ratio
        # self.grasp_thresh_bottom = self.grasp_open_val + (self.grasp_close_val - self.grasp_open_val)*self.grasp_thresh_ratio
        # self.grasp_thresh_upper = self.grasp_close_val - (self.grasp_close_val - self.grasp_open_val)*self.grasp_thresh_ratio
        # self.grasp_assist_delta = 100
        
        self.beta = args.beta
        
    def set_init_joint_state(self, init_exptime=5, init_freq=100):
        # init_arm_pos = np.array([2059, 2062, 2039, 1283, 3303, 2030, 1048, 2064,  294])
        # init_arm_pos = np.array([2036.0, 2076.0, 2032.0, 1027.0, 3032.0, 2046.0, 963.0, 2072.0, 258.0])
        # init_arm_pos = np.array([2038, 2044, 2043, 1739, 3207, 2041, 1764, 2034,  273])
        init_arm_pos = np.array([2342.5   , 1095.1666, 2372.4167, 1322.1666, 2344.2083, 3058.75  ,
       1312.8334, 1968.7916,  369.    ])
        # tar1_arm_pos = np.concatenate([self.right_arm_state[:-5], init_arm_pos[-5:]])        
        tar2_arm_pos = init_arm_pos
        
        # input("is it okey to move to init pos?")
        # self.move_arm(tar_arm_pos=tar1_arm_pos,
        #               arm_msg=self.right_arm_msg,
        #               arm_pub=self.right_arm_pub,
        #               exp_time=init_exptime,
        #               freq=init_freq)
        
        input("finish pos1, move pos2")
        self.move_arm(tar_arm_pos=tar2_arm_pos,
                      arm_msg=self.right_arm_msg,
                      arm_pub=self.right_arm_pub,
                      exp_time=init_exptime,
                      freq=init_freq)
    
    def move_arm(self, 
                tar_arm_pos, arm_msg, arm_pub,
                exp_time=3, freq=100,
                interpolate_type="spline"):
        nloop = exp_time * freq

        tar_arm_pos = np.clip(tar_arm_pos, self.vec_bounds[0], self.vec_bounds[1])
        curr_arm_pos = np.array(arm_msg.position)
        
        if interpolate_type == "linear":
            arm_traj = np.linspace(curr_arm_pos, tar_arm_pos , nloop)
        elif interpolate_type == "spline":
            time_points = np.array([0, exp_time])  # 開始時間と終了時間
            traj_points = np.vstack([curr_arm_pos, tar_arm_pos])
            splines = [
                CubicSpline(time_points, traj_points[:, i], bc_type=((1, 0), (1, 0)))
                for i in range(traj_points.shape[1])
            ]
            time_stamps = np.linspace(0, exp_time, nloop)
            arm_traj = np.array([spline(time_stamps) for spline in splines]).T
            
        for i in range(nloop):        
            arm_msg.header.stamp = rospy.Time.now()
            arm_msg.position = arm_traj[i]
            arm_pub.publish(arm_msg)
            time.sleep(1./freq)    
    
    
    
    def cv2_to_imgmsg(self, out_img, encoding="bgr8"):
        # 画像の幅、高さ、チャンネル数を取得
        height, width, channels = out_img.shape
        
        # Imageメッセージの作成
        img_msg = Image()
        img_msg.header = Header()
        img_msg.height = height
        img_msg.width = width
        img_msg.encoding = encoding
        img_msg.is_bigendian = False
        img_msg.step = channels * width  # 1行のバイト数
        img_msg.data = out_img.tobytes()  # NumPy配列をバイト列に変換
        
        return img_msg
    
    
    
    def visualize_channels_horizontal(self, image, padding=5):
        h, w, c = image.shape
        images = []

        for i in range(c):
            channel = image[:, :, i]
            # 修正点：np.ptp(channel) を使用
            norm = (channel - channel.min()) / (np.ptp(channel) + 1e-5)
            channel_img = (norm * 255).astype(np.uint8)
            channel_img = np.stack([channel_img] * 3, axis=2)
            images.append(channel_img)

        pad = 255 * np.ones((h, padding, 3), dtype=np.uint8)

        concatenated = images[0]
        for img in images[1:]:
            concatenated = np.concatenate((concatenated, pad, img), axis=1)

        return concatenated
    
    
    def run(self):
        if self.right_arm_state is not None:
            self.set_init_joint_state()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir_path = f"../output/{timestamp}"
            os.makedirs(output_dir_path)

            input("Are you ready?")
            # state_dict = {"key": None, "vec": None, "union": None}
            state = None
            y_vec_hat_list, y_img1_hat_list = [], []
            pred_vec_list, curr_vec_list = [], []
            prev_sc_enc_map_dict = None
            
            for loop_ct in range(self.nloop):
                right_img = cv2.convertScaleAbs(self.right_img, alpha=1, beta=self.beta)
                # right_img = self.right_img
                
                rt_right_eye_imgs = np.expand_dims(right_img, axis=[0,1])
                rt_vecs = np.expand_dims(np.clip(self.right_arm_state, 
                                                 self.vec_bounds[0], 
                                                 self.vec_bounds[1]), axis=[0,1])
                
                _rt_right_eye_imgs = self.processor.process_img(rt_right_eye_imgs, img_size=self.eye_img_size)
                _rt_vecs = self.processor.process_vec(rt_vecs)
                
                x_eye_img1 = torch.Tensor(_rt_right_eye_imgs[:,0]).to(torch.float32)
                x_vec = torch.Tensor(_rt_vecs[:,0]).to(torch.float32)
                
                if loop_ct > 0:
                    # x_eye_img1 = self.open_ratio * x_eye_img1 + (1.0 - self.open_ratio) * y_img1_hat_list[-1]
                    x_vec = self.open_ratio * x_vec + (1.0 - self.open_ratio) * y_vec_hat_list[-1]
                    
                (y_eye_img1_hat, y_vec_hat, # y_delta_vec_hat,
                 sc_enc_eye_key1, sc_dec_eye_key1, 
                 sm_enc_eye_key1, sm_dec_eye_key1, 
                 sc_enc_map_dict, sm_enc_map, delta_map,
                 state) = self.model(x_eye_img1, 
                                          prev_sc_enc_map_dict,
                                          x_vec, 
                                          state) # , x_vec
                
                # y_vec_hat = y_delta_vec_hat + x_vec
                prev_sc_enc_map_dict = sc_enc_map_dict
                
                y_vec_hat_list.append(y_vec_hat)
                y_img1_hat_list.append(y_eye_img1_hat)
                
                y_eye_img1_hat = y_eye_img1_hat.unsqueeze(dim=1).clip(0,1)
                y_vec_hat = y_vec_hat.unsqueeze(dim=1)
                
                sc_enc_eye_key1 = sc_enc_eye_key1.unsqueeze(dim=1)
                sc_dec_eye_key1 = sc_dec_eye_key1.unsqueeze(dim=1)
                sm_enc_eye_key1 = sm_enc_eye_key1.unsqueeze(dim=1)
                sm_dec_eye_key1 = sm_dec_eye_key1.unsqueeze(dim=1)
                
                sm_enc_map = sm_enc_map.unsqueeze(dim=1)
                delta_map = delta_map.unsqueeze(dim=1)
                
                sc_enc_eye_key1 = self.deprocessor.deprocess_key(sc_enc_eye_key1, self.eye_img_size)[0,0]
                sc_dec_eye_key1 = self.deprocessor.deprocess_key(sc_dec_eye_key1, self.eye_img_size)[0,0]
                sm_enc_eye_key1 = self.deprocessor.deprocess_key(sm_enc_eye_key1, self.eye_img_size)[0,0]
                sm_dec_eye_key1 = self.deprocessor.deprocess_key(sm_dec_eye_key1, self.eye_img_size)[0,0]
                
                pred_eye_img1 = self.deprocessor.deprocess_img(y_eye_img1_hat)[0,0].copy()
                pred_vec = self.deprocessor.deprocess_vec(y_vec_hat)[0,0].copy()
                pred_vec_list.append(pred_vec)
                
                sm_enc_map = self.deprocessor.deprocess_feat(sm_enc_map)[0,0].copy()
                delta_map = self.deprocessor.deprocess_feat(delta_map)[0,0].copy()
                
                curr_eye_img1 = self.deprocessor.deprocess_img(x_eye_img1.unsqueeze(1))[0,0].copy()
                curr_vec = self.deprocessor.deprocess_vec(x_vec.unsqueeze(1))[0,0].copy()
                curr_vec_list.append(curr_vec)
                                
                for i in range(self.key_dim):
                    cv2.circle(curr_eye_img1, tuple(sc_enc_eye_key1[i]), 1, (255,0,0), thickness=-1)
                    cv2.circle(curr_eye_img1, tuple(sc_dec_eye_key1[i]), 1, (0,0,255), thickness=-1)
                    cv2.circle(curr_eye_img1, tuple(sm_enc_eye_key1[i]), 1, (255,128,128), thickness=-1)
                    cv2.circle(curr_eye_img1, tuple(sm_dec_eye_key1[i]), 1, (128,128,255), thickness=-1)
                
                out_img = np.concatenate((curr_eye_img1, pred_eye_img1), axis=1)
                out_img_msg = self.cv2_to_imgmsg(out_img)
                self.pred_img_pub.publish(out_img_msg)
                
                out_sm_feat = self.visualize_channels_horizontal(sm_enc_map)
                out_sm_feat_msg = self.cv2_to_imgmsg(out_sm_feat)
                self.sm_feat_pub.publish(out_sm_feat_msg)

                out_delta_map = self.visualize_channels_horizontal(delta_map)

                self.right_arm_msg.header.stamp = rospy.Time.now()
                tar_right_pos = np.clip(np.round(pred_vec), self.vec_bounds[0], self.vec_bounds[1])
                
                # plt.figure()
                # plt.imshow(out_img)
                # plt.savefig(f"{output_dir_path}/out_img{loop_ct}.png")
                # plt.close()
                cv2.imwrite(f"{output_dir_path}/out_img{loop_ct}.png", out_img)

                # plt.figure()
                # plt.imshow(out_delta_map)
                # plt.savefig(f"{output_dir_path}/out_delta_map{loop_ct}.png")
                # plt.close()

                self.right_arm_msg.position = tar_right_pos
                if loop_ct > 10:
                    # pass
                    self.right_arm_pub.publish(self.right_arm_msg)
                
                # print(loop_ct)
                print(f"{loop_ct}: pred: {np.round(pred_vec)[-1]}")
                self.r.sleep()
            
            curr_vecs = (np.array(curr_vec_list)[1:] - 2048.0)/2048.0 * 180.0
            pred_vecs = (np.array(pred_vec_list)[:-1] - 2048.0)/2048.0 * 180.0
            abs_vecs = np.abs(curr_vecs - pred_vecs)
            
            json_data = {
                "curr_vecs": curr_vecs.tolist(),
                "pred_vecs": pred_vecs.tolist(),
                "abs_vecs": abs_vecs.tolist()
            }
            with open(f"{output_dir_path}/curr_vs_pred_err.json", "w") as f:
                json.dump(json_data, f, indent=2)
            
            plt.figure()
            for i in range(9):
                plt.plot(pred_vecs[:,i], label=f"joint{i}") # , label=f"joint{i}"
                plt.plot(curr_vecs[:,i], linestyle='--')
            plt.legend()
            plt.ylabel("deg")
            plt.savefig(f"{output_dir_path}/curr_pred_vec_trend.png")
            plt.close()
            
            plt.figure()
            for i in range(9):
                plt.plot(abs_vecs)
            plt.legend()
            plt.ylabel("deg")
            plt.savefig(f"{output_dir_path}/curr_pred_abs_vec_trend.png")
            plt.close()
            ipdb.set_trace()
            
            
        else:
            print("check joint publish!")
            exit()

if __name__ == "__main__":
    # How to use
    # move to tools folder and execute following command.
    # python3 playback.py    




    
    parser = argparse.ArgumentParser()
    # parser.add_argument("filename", type=str)
    parser.add_argument("--log_dir_name", type=str, default="20250711_2051_18")
    parser.add_argument("--freq", type=int, default=10)
    parser.add_argument("--exptime", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--open_ratio", type=float, default=1.0)
    # parser.add_argument("--grasp_thresh_ratio", type=float, default=0.3)
    parser.add_argument("--beta", type=int, default=20)
    parser.add_argument("--ckpt_idx", type=int, default=-1)
    args = parser.parse_args()
    
    try:
        rospy.init_node("task_node", anonymous=True)
        task = RTControl(args)
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
