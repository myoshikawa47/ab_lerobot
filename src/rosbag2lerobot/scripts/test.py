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

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.scripts.visualize_dataset import EpisodeSampler
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_folder", type=str, default=None)
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--webp", action="store_true")
args = parser.parse_args()



# Create a directory to store the video of the evaluation
output_directory = Path("outputs/test/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = 'cpu' if args.device < 0 else f"cuda:{args.device}"

# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
# pretrained_policy_path = "lerobot/diffusion_pusht"
# OR a path to a local outputs/train folder.
pretrained_policy_path = args.ckpt_folder

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.to(device)

# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)

# Reset the policy and environments to prepare for rollout
policy.reset()

# Load data
dataset = LeRobotDataset('null', root=args.data_dir)
start = dataset.episode_data_index['from'][args.idx].item()
end = dataset.episode_data_index['to'][args.idx].item()
frames = [{k: v.to(device) for k, v in dataset[i].items() if type(v)==torch.Tensor} for i in range(start, end)]
nloop = len(frames)

def save_gt(frame):
    return

# Inference
actions_list = []
key_list = []
state = None
for loop_ct in range(nloop):
    frame = frames[loop_ct]
    save_gt(frame)

    # prediction
    action = policy.select_action(frame)
    actions_list.append(action.cpu())

    print("loop_ct:{}, joint:{}".format(loop_ct, actions_list[-1]))

actions_np = np.array(actions_list)

# plot images
fig, ax = plt.subplots(1, 4, figsize=(24, 6), dpi=60)
ax = ax.flatten()

gt = dict()
gt = {k: np.array([frame[k].cpu() for frame in frames]) for k in frames[0].keys()}

def anim_update(i):
    for j in range(2):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(np.transpose(gt['observation.image.head.left'][i], (1, 2, 0)))
    ax[0].axis("off")
    ax[0].set_title("observation.image.head.left")

    # plot camera image
    ax[1].imshow(np.transpose(gt['observation.image.arm.right'][i], (1, 2, 0)))
    ax[1].axis("off")
    ax[1].set_title("observation.image.arm.right")

    # plot joint angle
    ax[2].set_xlim(0, nloop)
    ax[2].plot(gt['action'][:, :9], linestyle="dashed", c="k")
    for joint_idx in range(9):
        ax[2].plot(np.arange(i + 1), actions_np[: i + 1, joint_idx], c=f"C{joint_idx}")
    ax[2].set_xlabel("Step")
    ax[2].set_title("Joint angles")
    
    # plot base angle
    ax[3].set_xlim(0, nloop)
    ax[3].plot(gt['action'][:, 9:], linestyle="dashed", c="k")
    for joint_idx in range(9, 12):
        ax[3].plot(np.arange(i + 1), actions_np[: i + 1, joint_idx], c=f"C{joint_idx}")
    ax[3].set_xlabel("Step")
    ax[3].set_title("Base angles")


ani = anim.FuncAnimation(fig, anim_update, frames=nloop)
if args.webp:
    ani.save(os.path.join(output_directory, "debug_{}.webp".format(args.idx)), writer="ffmpeg", fps=10)
else:
    ani.save(os.path.join(output_directory, "debug_{}.mp4".format(args.idx)), fps=10, writer="ffmpeg")
