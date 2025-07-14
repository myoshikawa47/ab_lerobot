#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pylab as plt
from utils import resize_img, calc_minmax, list_to_numpy


def load_data(dir):
    steer_joints = []
    wheel_vels = []
    # left_joints = []
    right_joints = []
    right_tip_pos = []
    right_tip_quat = []
    right_currents = []
    right_taxels = []
    # head_l_images = []
    head_r_images = []
    hand_r_images = []
    # seq_length = []

    files = glob.glob(os.path.join(dir, "*.npz"))
    files.sort()

    for k, filename in enumerate(files):
        print(filename)
        npz_data = np.load(filename)
        # head_l_images.append(resize_img(npz_data["head_l_images"], (64, 64)))
        head_r_images.append(resize_img(npz_data["head_r_images"], (64, 64)))
        hand_r_images.append(resize_img(npz_data["hand_r_images"], (64, 64)))
        # head_l_images.append(resize_img(npz_data["head_l_images"], (256, 256)))
        # head_r_images.append(resize_img(npz_data["head_r_images"], (256, 256)))
        # hand_r_images.append(resize_img(npz_data["hand_r_images"], (256, 256)))
        _steer_joints = npz_data["steer_joints"]
        steer_joints.append(_steer_joints)
        _wheel_vels = npz_data["wheel_vels"]
        wheel_vels.append(_wheel_vels)
        # _left_joints = npz_data["left_joints"]
        # left_joints.append(_left_joints)
        _right_joints = npz_data["right_joints"]
        right_joints.append(_right_joints)       
        _right_tip_pos = npz_data["right_tip_pos"]
        right_tip_pos.append(_right_tip_pos)       
        _right_tip_quat = npz_data["right_tip_quat"]
        right_tip_quat.append(_right_tip_quat)       
        _right_currents = npz_data["right_currents"]
        right_currents.append(_right_currents)       
        _right_taxels = npz_data["right_taxels"]
        right_taxels.append(_right_taxels)       
        print(len(_steer_joints))

    # import pdb;pdb.set_trace()
    steer_joints = np.array(steer_joints, dtype=np.float32)
    wheel_vels = np.array(wheel_vels, dtype=np.float32)
    # left_joints = np.array(left_joints, dtype=np.float32)
    right_joints = np.array(right_joints, dtype=np.float32)
    right_tip_pos = np.array(right_tip_pos, dtype=np.float32)
    right_tip_quat = np.array(right_tip_quat, dtype=np.float32)
    right_currents = np.array(right_currents, dtype=np.float32)
    right_taxels = np.array(right_taxels, dtype=np.float32)
    # head_l_images = np.array(head_l_images, dtype=np.uint8)
    head_r_images = np.array(head_r_images, dtype=np.uint8)
    hand_r_images = np.array(hand_r_images, dtype=np.uint8)

    return head_r_images, hand_r_images, steer_joints, wheel_vels, right_joints, right_tip_pos, right_tip_quat, right_currents, right_taxels
    # return head_l_images, head_r_images, hand_r_images, steer_joints, wheel_vels, right_joints, right_currents


if __name__ == "__main__":
    
    # 0618 success, retry
    train_list = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 55, 57, 58, 59]
    test_list = [3, 8, 15, 17, 21, 28, 33, 42, 49, 56]

    # load data
    head_r_images, hand_r_images, steer_joints, wheel_vels, right_joints, right_tip_pos, right_tip_quat, right_currents, right_taxels = load_data("./bag/20250618_sentakuki_test_all")
    # head_r_images, hand_r_images, steer_joints, wheel_vels, right_joints, right_tip_pos, right_tip_quat, right_currents, right_taxels = load_data("./bag/20250618_sentakuki_test_retry")
    # head_r_images, hand_r_images, steer_joints, wheel_vels, right_joints, right_tip_pos, right_tip_quat, right_currents, right_taxels = load_data("./bag/20250618_sentakuki_test_success")
    # head_l_images, head_r_images, hand_r_images, steer_joints, wheel_vels, right_joints, right_currents = load_data("./bag/20250521_tohoku_denryoku_door")

    # save images and joints
    # np.save("./data/train/head_l_images.npy", head_l_images[train_list].astype(np.uint8))
    np.save("./data/train/head_r_images.npy", head_r_images[train_list].astype(np.uint8))
    np.save("./data/train/hand_r_images.npy", hand_r_images[train_list].astype(np.uint8))
    np.save("./data/train/steer_joints.npy", steer_joints[train_list].astype(np.float32))
    np.save("./data/train/wheel_vels.npy", wheel_vels[train_list].astype(np.float32))
    # np.save("./data/train/left_joints.npy", left_joints[train_list].astype(np.float32))
    np.save("./data/train/right_joints.npy", right_joints[train_list].astype(np.float32))
    np.save("./data/train/right_tip_pos.npy", right_tip_pos[train_list].astype(np.float32))
    np.save("./data/train/right_tip_quat.npy", right_tip_quat[train_list].astype(np.float32))
    np.save("./data/train/right_currents.npy", right_currents[train_list].astype(np.float32))
    np.save("./data/train/right_taxels.npy", right_taxels[train_list].astype(np.float32))
 
    # np.save("./data/test/head_l_images.npy", head_l_images[test_list].astype(np.uint8))
    np.save("./data/test/head_r_images.npy", head_r_images[test_list].astype(np.uint8))
    np.save("./data/test/hand_r_images.npy", hand_r_images[test_list].astype(np.uint8))
    np.save("./data/test/steer_joints.npy", steer_joints[test_list].astype(np.float32))
    np.save("./data/test/wheel_vels.npy", wheel_vels[test_list].astype(np.float32))
    # np.save("./data/test/left_joints.npy", left_joints[test_list].astype(np.float32))
    np.save("./data/test/right_joints.npy", right_joints[test_list].astype(np.float32))
    np.save("./data/test/right_tip_pos.npy", right_tip_pos[test_list].astype(np.float32))
    np.save("./data/test/right_tip_quat.npy", right_tip_quat[test_list].astype(np.float32))
    np.save("./data/test/right_currents.npy", right_currents[test_list].astype(np.float32))
    np.save("./data/test/right_taxels.npy", right_taxels[test_list].astype(np.float32))

    # save joint bounds
    steer_joint_bounds = calc_minmax(steer_joints)
    wheel_vel_bounds = calc_minmax(wheel_vels)
    # left_joint_bounds = calc_minmax(left_joints)
    right_joint_bounds = calc_minmax(right_joints)
    right_tip_pos_bounds = calc_minmax(right_tip_pos)
    right_tip_quat_bounds = calc_minmax(right_tip_quat)
    right_current_bounds = calc_minmax(right_currents)
    right_taxel_bounds = calc_minmax(right_taxels)

    np.save("./data/steer_joint_bounds.npy", steer_joint_bounds)
    np.save("./data/wheel_vel_bounds.npy", wheel_vel_bounds)
    # np.save("./data/left_joint_bounds.npy", left_joint_bounds)
    np.save("./data/right_joint_bounds.npy", right_joint_bounds)
    np.save("./data/right_tip_pos_bounds.npy", right_tip_pos_bounds)
    np.save("./data/right_tip_quat_bounds.npy", right_tip_quat_bounds)
    np.save("./data/right_current_bounds.npy", right_current_bounds)
    np.save("./data/right_taxel_bounds.npy", right_taxel_bounds)
