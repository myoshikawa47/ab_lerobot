import time
import os
import sys
import cv2
import glob
import argparse
import numpy as np
import torch

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.serde import deserialize_cdr, ros1_to_cdr

# typestore = get_typestore(Stores.LATEST)
typestore = get_typestore(Stores.ROS1_NOETIC)

# define custom msgs
from pathlib import Path
from rosbags.typesys import get_types_from_idl, get_types_from_msg

# from lerobot.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
sys.path.append('./scripts')
from features import airec_basic_features


def main(args):
    
    # initialize lerobot dataset
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.lerobot_dir,
        robot_type='airec_basic',
        features=airec_basic_features,
        tolerance_s=0.5,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    episode_task = 'debug'

    files = glob.glob(args.rosbag_dir+'/*.bag')
    files.sort()

    # Get the topics in the rosbag file
    # topics = [
    #     "/maharo1/lowerbody/command_states",
    #     # "/maharo1/left_arm/upperbody/online_joint_states",
    #     "/maharo1/right_arm/upperbody/online_joint_states",
    #     "/maharo1/right_arm/upperbody/joint_states",
    #     # "/l_hand_usb_cam/image_raw/compressed",
    #     "/r_hand_usb_cam/image_raw/compressed",
    #     "/zed2i/zed_node/stereo_raw/image_raw_color/compressed",
    #     # "/xServTopic"
    # ]

    topics = [
        '/l_hand_usb_cam/image_raw/compressed',
        # '/left_audio/audio_stamped',
        '/maharo/upperbody_head/joint_states',
        '/maharo/upperbody_head/online_joint_states',
        '/maharo1/left_arm/upperbody/joint_states',
        '/maharo1/left_arm/upperbody/online_joint_states',
        # '/maharo1/left_arm/upperbody/tip_pose',
        '/maharo1/left_arm/upperbody/taxtile',
        '/maharo1/lowerbody/command_states',
        '/maharo1/lowerbody/joint_states',
        '/maharo1/lowerbody/online_joint_states',
        # '/maharo1/lowerbody/spacenav/twist',
        # '/maharo1/torso/mode',
        '/maharo1/right_arm/upperbody/joint_states',
        '/maharo1/right_arm/upperbody/online_joint_states',
        # '/maharo1/right_arm/upperbody/tip_pose',
        '/maharo1/right_arm/upperbody/taxtile',
        '/r_hand_usb_cam/image_raw/compressed',
        # '/right_audio/audio_stamped',
        '/teleop/head_joint_states',
        '/zed2i/zed_node/stereo/image_rect_color/compressed',
        '/zed2i/zed_node/stereo_raw/image_raw_color/compressed',
        ]

    for file in files:
        print(file)
        frames = []
        last_frame = {
            'observation.image.head.left': None,
            'observation.image.head.right': None,
            'observation.image.arm.left': None,
            'observation.image.arm.right': None,
            'observation.state.head': None,
            'observation.state.left_arm.pos': None,
            'observation.state.right_arm.pos': None,
            'observation.state.base': None,
            'observation.state.left_arm.effort': None,
            'observation.state.right_arm.effort': None,
            'observation.state.left_arm.tactile': None,
            'observation.state.right_arm.tactile': None,
            'action.head': None,
            'action.left_arm': None,
            'action.right_arm': None,
            'action.base': None,
            }

        # Open the rosbag file
        with Reader(file) as reader:
            # Get the start and end times of the rosbag file
            start_time = reader.start_time # nanosecond
            end_time = reader.end_time

            # Set initial current time
            current_time = start_time

            # Loop through the rosbag file at regular intervals (args.freq)
            period = 1.0 / float(args.fps)
            nano_period = int(period * 10**9)
            # Set connections
            connections = []
            for topic in topics:
                connections.append([x for x in reader.connections if x.topic == topic]) 
            # Logging loop
            while current_time < end_time:
                current_frame = last_frame.copy()
                
                # Get the messages for each topic at the current time
                for connection in connections:
                    for connection, timestamp, rawdata in reader.messages(connections=connection, start=current_time): 
                        if timestamp >= current_time:
                            # observation.image
                            if connection.topic == "/zed2i/zed_node/stereo_raw/image_raw_color/compressed":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                np_arr = np.frombuffer(msg.data, np.uint8)
                                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                current_frame['observation.image.head.left'] = np_img[:, :int(np_img.shape[1]/2)].astype(np.uint8) # 360*640
                                # current_frame['observation.image.head.left'] = cv2.resize(np_img[:, :int(np_img.shape[1]/2)], (64, 36)).astype(np.uint8)
                                current_frame['observation.image.head.right'] = np_img[:, int(np_img.shape[1]/2):].astype(np.uint8) # 360*640
                                # current_frame['observation.image.head.right'] = cv2.resize(np_img[:, :int(np_img.shape[1]/2)], (64, 36)).astype(np.uint8)

                            if connection.topic == "/l_hand_usb_cam/image_raw/compressed":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                np_arr = np.frombuffer(msg.data, np.uint8)
                                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                import ipdb;ipdb.set_trace()
                                current_frame['observation.image.arm.left'] = cv2.rotate(np_img[:360].astype(np.uint8), cv2.ROTATE_180) # all images have to be the same shape
                                # current_frame['observation.image.arm.right'] = cv2.resize(cv2.rotate(np_img[:360].astype(np.uint8), cv2.ROTATE_180), (64, 36)) # all images have to be the same shape
                            
                            if connection.topic == "/r_hand_usb_cam/image_raw/compressed":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                np_arr = np.frombuffer(msg.data, np.uint8)
                                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                import ipdb;ipdb.set_trace()
                                current_frame['observation.image.arm.right'] = cv2.rotate(np_img[:360].astype(np.uint8), cv2.ROTATE_180) # all images have to be the same shape
                                # current_frame['observation.image.arm.right'] = cv2.resize(cv2.rotate(np_img[:360].astype(np.uint8), cv2.ROTATE_180), (64, 36)) # all images have to be the same shape
                            
                            # observation.state
                            if connection.topic == "/maharo/upperbody_head/online_joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                import ipdb;ipdb.set_trace()
                                current_frame['observation.state.head'] = msg.position
                                
                            if connection.topic == "/maharo1/left_arm/upperbody/online_joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['observation.state.left_arm.pos'] = msg.position
                                
                            if connection.topic == "/maharo1/right_arm/upperbody/online_joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['observation.state.right_arm.pos'] = msg.position
                                
                            if connection.topic == "/maharo1/lowerbody/online_joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['observation.state.base'] = msg.position
                                
                            # effort
                            if connection.topic == "/maharo1/left_arm/upperbody/joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['observation.state.left_arm.effort'] = msg.effort

                            if connection.topic == "/maharo1/right_arm/upperbody/joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['observation.state.right_arm.effort'] = msg.effort

                            # tactile (?)
                            if connection.topic == "/maharo1/left_arm/upperbody/taxtile":
                                # import ipdb;ipdb.set_trace()
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['observation.state.left_arm.tactile'] = msg.effort
                            
                            if connection.topic == "/maharo1/left_arm/upperbody/taxtile":
                                # import ipdb;ipdb.set_trace()
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['observation.state.right_arm.tactile'] = msg.effort

                            # action
                            if connection.topic == "/teleop/head_joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['action.head'] = msg.position

                            if connection.topic == "/maharo1/lowerbody/command_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                current_frame['action.base'] = msg.position
                            # リーダのjoint stateをactionとしたほうが良いかも
                            
                            break
                
                # if 'action.base' not in current_frame.keys():
                #     current_frame['action.base'] = np.zeros(3)
                print('============================SKIP============================')
                # import ipdb;ipdb.set_trace()
                # current_frame['action'] = np.concatenate([current_frame['action.right_arm'], current_frame['action.base']])
                # current_frame = {k: v for k, v in current_frame.items() if k in ['observation.image.head.left', 'observation.image.arm.right', 'observation.state', 'action']}
                
                # current_frame['next.done'] = torch.tensor(False, dtype=torch.bool)
                frames.append(current_frame)
                last_frame = current_frame

                # To the next time step
                current_time += nano_period
            
            # frames[-1]['next.done'] = torch.tensor(True, dtype=torch.bool)
            for frame in frames:
                dataset.add_frame(frame, task=episode_task)

            dataset.save_episode()

    # dataset.consolidate(keep_image_files=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_dir", type=str, default='./data/rosbag/test')
    parser.add_argument("--lerobot_dir", type=str, default='./data/lerobot_dataset/test')
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()

    main(args)