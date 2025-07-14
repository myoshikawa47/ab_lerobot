#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from utils import normalization

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

idx = int(args.idx)
# left_joints = np.load("./data/test/left_joints.npy")
right_joints = np.load("./data/test/right_joints.npy")
right_tip_pos = np.load("./data/test/right_tip_pos.npy")
right_tip_quat = np.load("./data/test/right_tip_quat.npy")
right_currents = np.load("./data/test/right_currents.npy")
right_taxels = np.load("./data/test/right_taxels.npy")
steer_joints = np.load("./data/test/steer_joints.npy")
wheel_vels = np.load("./data/test/wheel_vels.npy")

# left_joint_bounds = np.load("./data/left_joint_bounds.npy")
right_joint_bounds = np.load("./data/right_joint_bounds.npy")
right_tip_pos_bounds = np.load("./data/right_tip_pos_bounds.npy")
right_tip_quat_bounds = np.load("./data/right_tip_quat_bounds.npy")
right_current_bounds = np.load("./data/right_current_bounds.npy")
right_taxel_bounds = np.load("./data/right_taxel_bounds.npy")
steer_joint_bounds = np.load("./data/steer_joint_bounds.npy")
wheel_vel_bounds = np.load("./data/wheel_vel_bounds.npy")

# head_l_images = np.load("./data/test/head_l_images.npy")
head_r_images = np.load("./data/test/head_r_images.npy")
hand_r_images = np.load("./data/test/hand_r_images.npy")
N = head_r_images.shape[1]

# normalized joints
minmax = [0.1, 0.9]
# norm_left_joints = normalization(left_joints, left_joint_bounds, minmax)
norm_right_joints = normalization(right_joints, right_joint_bounds, minmax)
norm_right_tip_pos = normalization(right_tip_pos, right_tip_pos_bounds, minmax)
norm_right_tip_quat = normalization(right_tip_quat, right_tip_quat_bounds, minmax)
norm_right_currents = normalization(right_currents, right_current_bounds, minmax)
norm_right_taxels = normalization(right_taxels, right_taxel_bounds, minmax)
norm_steer_joints = normalization(steer_joints, steer_joint_bounds, minmax)
norm_wheel_vels = normalization(wheel_vels, wheel_vel_bounds, minmax)


# print data information
# print("load test data, index number is {}".format(idx))
# print(
#     "Left Joint: shape={}, min={:.3g}, max={:.3g}".format(
#         left_joints.shape, left_joints.min(), left_joints.max()
#     )
# )
# print(
#     "Norm Left Joint: shape={}, min={:.3g}, max={:.3g}".format(
#         norm_left_joints.shape, norm_left_joints.min(), norm_left_joints.max()
#     )
# )

# print(
#     "Right Joint: shape={}, min={:.3g}, max={:.3g}".format(
#         right_joints.shape, right_joints.min(), right_joints.max()
#     )
# )
# print(
#     "Norm Right Joint: shape={}, min={:.3g}, max={:.3g}".format(
#         norm_right_joints.shape, norm_right_joints.min(), norm_right_joints.max()
#     )
# )


# print(
#     "Steer Joint: shape={}, min={:.3g}, max={:.3g}".format(
#         steer_joints.shape, steer_joints.min(), steer_joints.max()
#     )
# )
# print(
#     "Norm Steer Joint: shape={}, min={:.3g}, max={:.3g}".format(
#         norm_steer_joints.shape, norm_steer_joints.min(), norm_steer_joints.max()
#     )
# )

# print(
#     "Wheel Velocities: shape={}, min={:.3g}, max={:.3g}".format(
#         wheel_vels.shape, wheel_vels.min(), wheel_vels.max()
#     )
# )
# print(
#     "Norm Wheel Velocities: shape={}, min={:.3g}, max={:.3g}".format(
#         norm_wheel_vels.shape, norm_wheel_vels.min(), norm_wheel_vels.max()
#     )
# )


# plot images and normalized values
fig, ax = plt.subplots(2, 4, figsize=(14, 6), dpi=60)


def anim_update(i):
    
    print(i)
    for j in range(2):
        for k in range(4):
            ax[j][k].cla()

    # plot head left image
    # ax[0][0].imshow(head_l_images[idx, i, :, :, ::-1])
    # ax[0][0].axis("off")
    # ax[0][0].set_title("Image")

    # plot head right image
    ax[0][0].imshow(head_r_images[idx, i, :, :, ::-1])
    ax[0][0].axis("off")
    ax[0][0].set_title("Image")

    # plot hand right image
    ax[1][0].imshow(hand_r_images[idx, i, :, :, ::-1])
    ax[1][0].axis("off")
    ax[1][0].set_title("Image")

    # plot normlized left joint angle
    # ax[0][1].set_ylim(0.0, 1.0)
    # ax[0][1].set_xlim(0, N)
    # ax[0][1].plot(norm_left_joints[idx], linestyle="dashed", c="k")

    # for joint_idx in range(len(left_joint_bounds[0])):
    #     ax[0][1].plot(np.arange(i + 1), norm_left_joints[idx, : i + 1, joint_idx])
    # ax[0][1].set_xlabel("Step")
    # ax[0][1].set_title("Normalized Left Joint")

    # plot normlized right joint current
    ax[0][1].set_ylim(0.0, 1.0)
    ax[0][1].set_xlim(0, N)
    ax[0][1].plot(norm_right_currents[idx], linestyle="dashed", c="k")

    for joint_idx in range(len(right_current_bounds[0])):
        ax[0][1].plot(np.arange(i + 1), norm_right_currents[idx, : i + 1, joint_idx])
    ax[0][1].set_xlabel("Step")
    ax[0][1].set_title("Normalized Right Current")

    # plot normalized leader joint angle
    ax[1][1].set_ylim(0.0, 1.0)
    ax[1][1].set_xlim(0, N)
    ax[1][1].plot(norm_right_joints[idx], linestyle="dashed", c="k")

    for joint_idx in range(len(right_joint_bounds[0])):
        ax[1][1].plot(np.arange(i + 1), norm_right_joints[idx, : i + 1, joint_idx])
    ax[1][1].set_xlabel("Step")
    ax[1][1].set_title("Normalized Right Joint")

    # plot normlized taxel
    ax[0][2].set_ylim(0.0, 1.0)
    ax[0][2].set_xlim(0, N)
    ax[0][2].plot(norm_right_taxels[idx], linestyle="dashed", c="k")

    for joint_idx in range(len(right_taxel_bounds[0])):
        ax[0][2].plot(np.arange(i + 1), norm_right_taxels[idx, : i + 1, joint_idx])
    ax[0][2].set_xlabel("Step")
    ax[0][2].set_title("Normalized Steer Joint")

    # plot normlized steer joint angle
    # ax[0][2].set_ylim(0.0, 1.0)
    # ax[0][2].set_xlim(0, N)
    # ax[0][2].plot(norm_steer_joints[idx], linestyle="dashed", c="k")

    # for joint_idx in range(len(steer_joint_bounds[0])):
    #     ax[0][2].plot(np.arange(i + 1), norm_steer_joints[idx, : i + 1, joint_idx])
    # ax[0][2].set_xlabel("Step")
    # ax[0][2].set_title("Normalized Steer Joint")

    # plot normlized wheel joint velocity
    ax[1][2].set_ylim(0.0, 1.0)
    ax[1][2].set_xlim(0, N)
    ax[1][2].plot(norm_wheel_vels[idx], linestyle="dashed", c="k")

    for joint_idx in range(len(wheel_vel_bounds[0])):
        ax[1][2].plot(np.arange(i + 1), norm_wheel_vels[idx, : i + 1, joint_idx])
    ax[1][2].set_xlabel("Step")
    ax[1][2].set_title("Normalized Wheel Velocity")

    # plot normlized tip position
    ax[0][3].set_ylim(0.0, 1.0)
    ax[0][3].set_xlim(0, N)
    ax[0][3].plot(norm_right_tip_pos[idx], linestyle="dashed", c="k")

    for joint_idx in range(len(right_tip_pos_bounds[0])):
        ax[0][3].plot(np.arange(i + 1), norm_right_tip_pos[idx, : i + 1, joint_idx])
    ax[0][3].set_xlabel("Step")
    ax[0][3].set_title("Normalized Tip Pos")


    # plot normlized tip quaternion
    ax[1][3].set_ylim(0.0, 1.0)
    ax[1][3].set_xlim(0, N)
    ax[1][3].plot(norm_right_tip_quat[idx], linestyle="dashed", c="k")

    for joint_idx in range(len(right_tip_quat_bounds[0])):
        ax[1][3].plot(np.arange(i + 1), norm_right_tip_quat[idx, : i + 1, joint_idx])
    ax[1][3].set_xlabel("Step")
    ax[1][3].set_title("Normalized Tip Quaternion")


ani = anim.FuncAnimation(fig, anim_update, interval=int(N / 10), frames=N)
ani.save("./output/check_data_{}.gif".format(idx))
