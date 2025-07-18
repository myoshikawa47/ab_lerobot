import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from pathlib import Path

import imageio
import torch

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)
from lerobot.scripts.visualize_dataset import EpisodeSampler


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--webp", action="store_true")
args = parser.parse_args()


# Create a directory to store the video of the evaluation
output_directory = Path(args.out_dir)
output_directory.mkdir(parents=True, exist_ok=True)

# Load data
dataset = LeRobotDataset('null', root=args.data_dir)
start = dataset.episode_data_index['from'][args.idx].item()
end = dataset.episode_data_index['to'][args.idx].item()
frames = [dataset[i] for i in range(start, end)]
nloop = len(frames)

# plot images
fig, ax = plt.subplots(2, 4, figsize=(24, 12), dpi=60)
ax = ax.flatten()

gt = dict()
gt = {k: np.array([frame[k].cpu() for frame in frames]) for k, v in frames[0].items() if type(v)==torch.Tensor}
# gt['observation.state.pos'] = np.concatenate([gt['observation.state.head'], gt['observation.state.left_arm.pos'], gt['observation.state.right_arm.pos']], axis=-1)
# gt['observation.state.torque'] = np.concatenate([gt['observation.state.left_arm.torque'], gt['observation.state.right_arm.torque']], axis=-1)
# gt['observation.state.tactile'] = np.concatenate([gt['observation.state.left_arm.tactile'], gt['observation.state.right_arm.tactile']], axis=-1)
# gt['action'] = np.concatenate([gt['action.head'], gt['action.left_arm'], gt['action.right_arm']], axis=-1)

def anim_update(i):
    print(i)
    for j in range(len(ax)):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(np.transpose(gt['observation.image.head.left'][i, ::-1, ::10, ::10], (1, 2, 0)))
    ax[0].axis("off")
    ax[0].set_title("observation.image.head.left")

    # plot camera image
    ax[1].imshow(np.transpose(gt['observation.image.head.right'][i, ::-1, ::10, ::10], (1, 2, 0)))
    ax[1].axis("off")
    ax[1].set_title("observation.image.head.right")

    # plot camera image
    ax[2].imshow(np.transpose(gt['observation.image.arm.left'][i, ::-1, ::10, ::10], (1, 2, 0)))
    ax[2].axis("off")
    ax[2].set_title("observation.image.arm.left")

    # plot camera image
    ax[3].imshow(np.transpose(gt['observation.image.arm.right'][i, ::-1, ::10, ::10], (1, 2, 0)))
    ax[3].axis("off")
    ax[3].set_title("observation.image.arm.right")

    # plot joint angle
    ax[4].set_xlim(0, nloop)
    # ax[4].plot(gt['observation.state.pos'], linestyle="dashed", c="k")
    ax[4].plot(gt['observation.state'], linestyle="dashed", c="k")
    for joint_idx in range(2+9*2):
        ax[4].plot(np.arange(i + 1), gt['action'][: i + 1, joint_idx], c=f"C{joint_idx}")
    ax[4].set_xlabel("Step")
    ax[4].set_title("state-action")
    
    # # plot joint angle
    # ax[5].set_xlim(0, nloop)
    # ax[5].plot(gt['observation.state.torque'], linestyle="dashed", c="k")
    # for joint_idx in range(9*2):
    #     ax[5].plot(np.arange(i + 1), gt['observation.state.torque'][: i + 1, joint_idx], c=f"C{joint_idx}")
    # ax[5].set_xlabel("Step")
    # ax[5].set_title("torque")
   
    # # plot joint angle
    # ax[6].set_xlim(0, nloop)
    # ax[6].plot(gt['observation.state.tactile'], linestyle="dashed", c="k")
    # for joint_idx in range(2*2):
    #     ax[6].plot(np.arange(i + 1), gt['observation.state.tactile'][: i + 1, joint_idx], c=f"C{joint_idx}")
    # ax[6].set_xlabel("Step")
    # ax[6].set_title("tactile")

    ax[7].axis("off")

ani = anim.FuncAnimation(fig, anim_update, frames=nloop)
savename = os.path.basename(args.data_dir)
if args.webp:
    ani.save(os.path.join(output_directory, "{}_{}.webp".format(savename, args.idx)), fps=10, writer="ffmpeg")
else:
    print('writing mp4')
    ani.save(os.path.join(output_directory, "debug_{}.mp4".format(savename, args.idx)), fps=10, writer="ffmpeg")