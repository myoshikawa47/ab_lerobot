import os
import sys
import json
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
    LeRobotDatasetMetadata,
    MultiLeRobotDataset,
)
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.scripts.visualize_dataset import EpisodeSampler
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
# from work.tmp.lerobot.policies.smolvla import smolvlm_with_expert

# from eipl.utils import restore_args, tensor2numpy, deprocess_img, normalization, resize_img


# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_folder", type=str, default=None)
parser.add_argument("--policy_type", choises=['smolvla', 'act', 'diffusion'], default='smolvla')
parser.add_argument("--repo_id", type=str, default='null')
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--savename", type=str, default=None)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--webp", action="store_true")
args = parser.parse_args()

device = 'cpu' if args.device < 0 else f"cuda:{args.device}"

# Load policy
pretrained_policy_path = args.ckpt_folder
# # adapt device in config.json
# config_path = os.path.join(pretrained_policy_path, "config.json")
# with open(config_path, "r") as f:
#     config = json.load(f)
# config["device"] = device
# with open(config_path, "w") as f:
#     json.dump(config, f)
if args.policy_type == 'smolvla':
    policy = SmolVLAPolicy.from_pretrained(pretrained_policy_path)
elif args.policy_type == 'act':
    policy = ACTPolicy.from_pretrained(pretrained_policy_path)
elif args.policy_type == 'diffusion':
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.reset()
policy.eval()
policy.to(device)

print(policy.config.input_features)
print(policy.config.output_features)


# Load data
# dataset = LeRobotDataset('null', root=args.data_dir) # older version
ds_meta = LeRobotDatasetMetadata(
    args.repo_id, root=args.data_dir,
)
delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
dataset = LeRobotDataset(
    args.repo_id,
    root=args.data_dir,
    delta_timestamps=delta_timestamps,
)
start = dataset.episode_data_index['from'][args.idx].item()
end = dataset.episode_data_index['to'][args.idx].item()
frames = [dataset[i] for i in range(start, end)]
for k, v in frames[0].items():
    if isinstance(v, torch.Tensor):
        for frame in frames: 
            frame[k] = frame[k].unsqueeze(0).to(device)
nloop = len(frames)


# Inference
actions_list = []
for loop_ct in range(nloop):
    frame = frames[loop_ct]

    # prediction
    action = policy.select_action(frame)
    actions_list.append(action.cpu())

    print("loop_ct:{}, joint:{}".format(loop_ct, actions_list[-1]))

actions_np = np.array(actions_list).reshape(nloop, -1)


# plot images
fig, ax = plt.subplots(1, 4, figsize=(24, 6), dpi=60)
ax = ax.flatten()

# ground truth
gt = dict()
gt = {k: np.stack([frame[k].squeeze(0).cpu().numpy() for frame in frames]) for k, v in frames[0].items() if type(v)==torch.Tensor}
# prepare dataset which has multiple inputs or outputs for animation
for k in gt.keys():
    if ('image' in k and len(gt[k].shape) == 5) \
        or ('observation.state' in k and len(gt[k].shape) == 3) \
        or ('action' in k and len(gt[k].shape) == 3):
            gt[k] = gt[k][:, 0]

def anim_update(i):
    print(i)
    for j in range(len(ax)):
        ax[j].cla()

    # # plot camera image
    # ax[0].imshow(np.transpose(gt['observation.image.head.left'][i], (1, 2, 0)))
    # ax[0].axis("off")
    # ax[0].set_title("observation.image.head.left")

    # plot camera image
    ax[1].imshow(np.transpose(gt['observation.image.head.right'][i, ::-1, ::10, ::10], (1, 2, 0)))
    ax[1].axis("off")
    ax[1].set_title("observation.image.head.right")

    # plot joint angle
    ax[2].set_xlim(0, nloop)
    ax[2].plot(gt['action'], linestyle="dashed", c="k")
    for joint_idx in range(gt['action'].shape[-1]):
        ax[2].plot(np.arange(i + 1), actions_np[: i + 1, joint_idx], c=f"C{joint_idx}")
    ax[2].set_xlabel("Step")
    ax[2].set_title("Joint angles")
    
    # # plot base angle
    # ax[3].set_xlim(0, nloop)
    # ax[3].plot(gt['action'][:, 9:], linestyle="dashed", c="k")
    # for joint_idx in range(9, 12):
    #     ax[3].plot(np.arange(i + 1), actions_np[: i + 1, joint_idx], c=f"C{joint_idx}")
    # ax[3].set_xlabel("Step")
    # ax[3].set_title("Base angles")


ani = anim.FuncAnimation(fig, anim_update, frames=nloop)
os.path.mkdir(args.out_dir)
savename = args.savename if args.savename else args.policy_type
if args.webp:
    ani.save(os.path.join(args.out_dir, "{}_{}.webp".format(savename, args.idx)), fps=10, writer="ffmpeg")
else:
    ani.save(os.path.join(args.out_dir, "{}_{}.mp4".format(savename, args.idx)), fps=10, writer="ffmpeg")
