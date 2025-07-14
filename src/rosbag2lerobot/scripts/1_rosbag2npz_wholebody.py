import time
import os
import cv2
import glob
import argparse
import numpy as np

# import PyKDL
# import kdl_parser_py.urdf as kdl_parser

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from rosbags.serde import deserialize_cdr, ros1_to_cdr

# typestore = get_typestore(Stores.LATEST)
typestore = get_typestore(Stores.ROS1_NOETIC)

# define custom msgs
from pathlib import Path
from rosbags.typesys import get_types_from_idl, get_types_from_msg

# maybe for tactile sensor of Xela
# add_types = {} # Plain dictionary to hold message definitions.

# msg_text = Path('./msg/Forces.msg').read_text() # Read definitions to python strings.
# add_types.update(get_types_from_msg(msg_text, 'xela_server_ros/msg/Forces')) # Add definitions from one msg file to the dict.
# msg_text = Path('./msg/Taxel.msg').read_text() # Read definitions to python strings.
# add_types.update(get_types_from_msg(msg_text, 'xela_server_ros/msg/Taxel')) # Add definitions from one msg file to the dict.
# msg_text = Path('./msg/SensorFull.msg').read_text() # Read definitions to python strings.
# add_types.update(get_types_from_msg(msg_text, 'xela_server_ros/msg/SensorFull')) # Add definitions from one msg file to the dict.
# msg_text = Path('./msg/SensStream.msg').read_text() # Read definitions to python strings.
# add_types.update(get_types_from_msg(msg_text, 'xela_server_ros/msg/SensStream')) # Add definitions from one msg file to the dict.

# typestore.register(add_types)


# Class of Gravity Compensator
class CalcKinematics():
    
    def __init__(self, URDF_path):
        
        # KDL chain
        self.chain_ = PyKDL.Chain()
        
        # KDL tree
        self.tree_ = PyKDL.Tree()

        # Parse URDF file to KDL tree
        ok, self.tree_ = kdl_parser.treeFromFile(URDF_path)

        if (not ok):
            print("[GravityCompensationController] Failed to parse URDF file")
        
        # Get chain
        self.chain_ = self.tree_.getChain("base_link", "tip_link")
        if (not ok):
            print("[GravityCompensationController] Could not get KDL chain from tree")

        # set nuber of joint + gripper joint
        self.joint_num_ = self.chain_.getNrOfJoints()
        self.end_id_ = self.joint_num_ + 2

        # Joint position
        self.joint_position_ = PyKDL.JntArray(self.joint_num_)

        # Jacobian matrix
        self.jacob_ = PyKDL.Jacobian(self.joint_num_)

        # Set Forward Kinematics Solver
        self.fk_kdl_ = PyKDL.ChainFkSolverPos_recursive(self.chain_)

        # Set Jacobian Solver
        self.jacob_kdl_ = PyKDL.ChainJntToJacSolver(self.chain_)
        
        
    # calculate forward kinematics
    def forward_kinematics(self, current_position, id):

       # Set current pos list[] to JntArray
        for i in range(self.joint_num_):
            self.joint_position_[i] = current_position[i]
        
        # Calc Forward Kinematics
        frame = PyKDL.Frame()
        res = self.fk_kdl_.JntToCart(self.joint_position_, frame, id)
        roll, pitch, yaw = frame.M.GetRPY()
        rotation = PyKDL.Rotation.RPY(roll, pitch, yaw)
        qx, qy, qz, qw = rotation.GetQuaternion()
    
        return np.array([frame.p.x(), frame.p.y(), frame.p.z()]), np.array([qx, qy, qz, qw])
        # return np.array([frame.p.x(), frame.p.y(), frame.p.z()]), np.array([roll, pitch, yaw])


    # calculate jacobian matrix
    def jacobian(self, current_position):

        # Set current pos list[] to JntArray
        for i in range(self.joint_num_):
            self.joint_position_[i] = current_position[i]
        
        # Calc Forward Kinematics
        self.jacob_kdl_.JntToJac(self.joint_position_, self.jacob_)
        jacob = self.kdl_to_mat(self.jacob_)
        
        return jacob

    def kdl_to_mat(self, data):
        mat = np.mat(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i, j] = data[i, j]
        return mat
    
    def joint_list_to_kdl(self, q):
        if q is None:
            return None
        if type(q) == np.matrix and q.shape[1] == 0:
            q = q.T.tolist()[0]
        q_kdl = PyKDL.JntArray(len(q))
        for i, q_i in enumerate(q):
            q_kdl[i] = q_i
        return q_kdl

def eq_hist(frame):
    _frame = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    _frame[:,:,0] = cv2.equalizeHist(_frame[:,:,0])
    frame_eqhist = cv2.cvtColor(_frame, cv2.COLOR_YUV2RGB)
    return frame_eqhist 

def list_extention(lst, target_length):
    # if not lst:
    #     raise ValueError("input list is empty")
    
    current_length = len(lst)
    if current_length >= target_length:
        return lst[:target_length]  # 既に所望の長さ以上の場合は切り詰める
    
    last_value = lst[-1]
    # 必要な分だけ最後の値を繰り返す
    extension = [last_value] * (target_length - current_length)
    return lst + extension

def main(args):
    files = glob.glob(args.bag_dir+'/*')
    files.sort()

    # urdf_path = "./urdf/maharo_follower_right.urdf"
    # fk = CalcKinematics(urdf_path)

    for file in files:
        print(file)
        # savename = file + ".npz"
        savename = os.path.splitext(file)[0] + ".npz"
        # Open the rosbag file
        with Reader(file) as reader:
            # Get the start and end times of the rosbag file
            start_time = reader.start_time # nanosecond
            end_time = reader.end_time

            # for connection in reader.connections:
            #     print(connection.topic, connection.msgtype)

            # import sys
            # sys.exit()

            # Get the topics in the rosbag file
            topics = [
                "/maharo1/lowerbody/command_states",
                # "/maharo1/left_arm/upperbody/online_joint_states",
                "/maharo1/right_arm/upperbody/online_joint_states",
                "/maharo1/right_arm/upperbody/joint_states",
                # "/l_hand_usb_cam/image_raw/compressed",
                "/r_hand_usb_cam/image_raw/compressed",
                "/zed2i/zed_node/stereo_raw/image_raw_color/compressed",
                # "/xServTopic"
            ]

            # Set initial current time
            current_time = start_time

            # Set blank lists
            # left_joint_list = []
            right_joint_list = []
            right_tip_pos_list = []
            right_tip_quat_list = []
            right_current_list = []
            right_taxel_list = []
            steer_joint_list = []
            wheel_vel_list = []
            # head_l_image_list = []
            head_r_image_list = []
            hand_r_image_list = []
            
            # Loop through the rosbag file at regular intervals (args.freq)
            period = 1.0 / float(args.freq)
            nano_period = int(period * 10**9)

            # Set connections
            connections = []
            for topic in topics:
                connections.append([x for x in reader.connections if x.topic == topic]) 
            
            # Logging loop
            while current_time < end_time:
                # print(current_time*10**-9)

                # Get the messages for each topic at the current time
                for connection in connections:
                    for connection, timestamp, rawdata in reader.messages(connections=connection, start=current_time): 
                        if timestamp >= current_time:

                            if connection.topic == "/maharo1/lowerbody/command_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                steer_joint_list.append(msg.position)
                                wheel_vel_list.append(msg.velocity[4:])
                                # import ipdb;ipdb.set_trace()
                            
                            # if connection.topic == "/maharo1/left_arm/upperbody/online_joint_states":
                            #     cdrdata = ros1_to_cdr(rawdata, connection.msgtype)
                            #     msg = deserialize_cdr(cdrdata, connection.msgtype)
                            #     left_joint_list.append(msg.position)

                            if connection.topic == "/xServTopic":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                sum_x = 0
                                sum_y = 0
                                sum_z = 0
                                for taxel in msg.sensors[0].taxels:
                                    sum_x = sum_x + taxel.x
                                    sum_y = sum_y + taxel.y
                                    sum_z = sum_z + taxel.z
                                tmp = [sum_x / 16.0, sum_y / 16.0, sum_z / 16.0]
                                # print("taxel sum : ", tmp)
                                right_taxel_list.append(tmp)
                            
                            if connection.topic == "/maharo1/right_arm/upperbody/online_joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                right_joint_list.append(msg.position)

                            if connection.topic == "/maharo1/right_arm/upperbody/joint_states":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                right_current_list.append(msg.effort)

                                # pos, quat = fk.forward_kinematics(msg.position, fk.end_id_)
                                # right_tip_pos_list.append(pos)
                                # right_tip_quat_list.append(quat)
                                                            
                            if connection.topic == "/zed2i/zed_node/stereo_raw/image_raw_color/compressed":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                np_arr = np.frombuffer(msg.data, np.uint8)
                                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                np_img = np_img[:, :]
                                # np_img_cut_r = np_img[170:320, 920:1070] # 0618 success
                                np_img_cut_r = np_img[120:370, 860:1110] # 0618 retry, all
                                #adjusted_image = cv2.convertScaleAbs(np_img, alpha=2.0, beta=50.0)
                                # adjusted_image = eq_hist(np_img_cut)                                
                                #image_list.append(adjusted_image[100:250,250:400].astype(np.uint8))
                                # head_l_image_list.append(np_img_cut_l.astype(np.uint8))
                                head_r_image_list.append(np_img_cut_r.astype(np.uint8))
                                # image_list.append(np_img_cut.astype(np.uint8))
                                
                            if connection.topic == "/r_hand_usb_cam/image_raw/compressed":
                                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                                np_arr = np.frombuffer(msg.data, np.uint8)
                                np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                np_img = np_img[:, :]
                                # np_img_cut = np_img[0:300, 50:350] # 0507
                                # np_img_cut = np_img[200:400, 150:350] # 0509
                                np_img_cut = np_img # 0514 task1
                                #adjusted_image = cv2.convertScaleAbs(np_img, alpha=2.0, beta=50.0)
                                adjusted_image = eq_hist(np_img_cut)                                
                                #image_list.append(adjusted_image[100:250,250:400].astype(np.uint8))
                                hand_r_image_list.append(cv2.rotate(np_img_cut.astype(np.uint8), cv2.ROTATE_180))
                                # image_list.append(np_img_cut.astype(np.uint8))
                                
                            break

                # To the next time step
                current_time += nano_period

        import ipdb;ipdb.set_trace()
        # Get shorter lenght
        # shorter_length = min(len(steer_joints), len(wheel_vels), len(right_joints), len(right_tip_pos), len(right_tip_quat), len(right_currents), len(right_taxels), len(head_r_images), len(hand_r_images))
        # print("hand_r_images: ", len(hand_r_images))
        # print("head_r_images: ", len(head_r_images))
        # print("head_l_images: ", len(head_l_images))
        # print("steer: ", len(steer_joints))
        # print("wheel: ", len(wheel_vels))
        # print("right joint: ", len(right_joints))
        # print("right current: ", len(right_currents))
        # shorter_length = 145
        shorter_length = 295
        # print("shorter_length : ", shorter_length)
        
        # Trim or Extention
        steer_joint_list = list_extention(steer_joint_list, shorter_length)
        wheel_vel_list = list_extention(wheel_vel_list, shorter_length)
        # left_joint_list = list_extention(left_joint_list, shorter_length)
        right_joint_list = list_extention(right_joint_list, shorter_length)
        # right_tip_pos_list = list_extention(right_tip_pos_list, shorter_length)
        # right_tip_quat_list = list_extention(right_tip_quat_list, shorter_length)
        right_current_list = list_extention(right_current_list, shorter_length)
        # right_taxel_list = list_extention(right_taxel_list, shorter_length)
        # head_l_image_list = list_extention(head_l_image_list, shorter_length)
        head_r_image_list = list_extention(head_r_image_list, shorter_length)
        hand_r_image_list = list_extention(hand_r_image_list, shorter_length)

        # Convert list to array
        steer_joints = np.array(steer_joint_list, dtype=np.float32)
        wheel_vels = np.array(wheel_vel_list, dtype=np.float32)
        # left_joints = np.array(left_joint_list, dtype=np.float32)
        right_joints = np.array(right_joint_list, dtype=np.float32)
        # right_tip_pos = np.array(right_tip_pos_list, dtype=np.float32)
        # right_tip_quat = np.array(right_tip_quat_list, dtype=np.float32)
        right_currents = np.array(right_current_list, dtype=np.float32)
        # right_taxels = np.array(right_taxel_list, dtype=np.float32)
        # head_l_images = np.array(head_l_image_list, dtype=np.uint8)
        head_r_images = np.array(head_r_image_list, dtype=np.uint8)
        hand_r_images = np.array(hand_r_image_list, dtype=np.uint8)

        # Save
        # np.savez(
        #     savename, steer_joints=steer_joints, wheel_vels=wheel_vels, right_joints=right_joints, right_tip_pos=right_tip_pos, right_tip_quat=right_tip_quat, right_currents=right_currents, right_taxels=right_taxels, head_r_images=head_r_images, hand_r_images=hand_r_images
        # )
        np.savez(
            savename, steer_joints=steer_joints, wheel_vels=wheel_vels, right_joints=right_joints, right_currents=right_currents, head_r_images=head_r_images, hand_r_images=hand_r_images
        )
        # np.savez(
        #     savename, steer_joints=steer_joints, wheel_vels=wheel_vels, left_joints=left_joints, right_joints=right_joints, right_currents=right_currents, head_l_images=head_l_images, head_r_images=head_r_images, hand_r_images=hand_r_images
        # )

        # Plot results
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # label_mov = ['w1','w2','w3']
        # label_arm = ['j1','j2','j3','j4','j5','j6','j7','j8','g']
        # label_xyz = ['X','Y','Z']
        # label_xyzw = ['X','Y','Z', "W"]
        # color = [i for i in mcolors.TABLEAU_COLORS.keys()]

        # fig = plt.figure()
        # ax1 = fig.add_subplot(231)
        # ax2 = fig.add_subplot(232)
        # ax3 = fig.add_subplot(233)
        # ax4 = fig.add_subplot(234)
        # ax5 = fig.add_subplot(235)
        # ax6 = fig.add_subplot(236)
        # # [ax1.plot(steer_joints[:,i], linestyle='solid', c=color[i], label=label_mov[i]) for i in range(steer_joints.shape[-1])]
        # [ax1.plot(wheel_vels[:,i], linestyle='solid', c=color[i], label=label_mov[i]) for i in range(wheel_vels.shape[-1])]
        # [ax2.plot(right_currents[:,i], linestyle='solid', c=color[i], label=label_arm[i]) for i in range(right_currents.shape[-1])]
        # # [ax3.plot(left_joints[:,i], linestyle='solid', c=color[i], label=label_arm[i]) for i in range(left_joints.shape[-1])]
        # [ax3.plot(right_joints[:,i], linestyle='solid', c=color[i], label=label_arm[i]) for i in range(right_joints.shape[-1])]
        # [ax4.plot(right_taxels[:,i], linestyle='solid', c=color[i], label=label_xyz[i]) for i in range(right_taxels.shape[-1])]
        # [ax5.plot(right_tip_pos[:,i], linestyle='solid', c=color[i], label=label_xyz[i]) for i in range(right_tip_pos.shape[-1])]
        # [ax6.plot(right_tip_quat[:,i], linestyle='solid', c=color[i], label=label_xyzw[i]) for i in range(right_tip_quat.shape[-1])]
        
        # plt.legend()
        # plt.show()

    
        # check_frame_step = 200
        # check_frame = 1
        # while check_frame < shorter_length:
        #     print("check_frame : ", check_frame)
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(211)
        #     ax2 = fig.add_subplot(212)
        #     # ax1 = fig.add_subplot(221)
        #     # ax2 = fig.add_subplot(222)
        #     # ax3 = fig.add_subplot(223)
        #     # ax4 = fig.add_subplot(224)
        #     # ax1.imshow(head_l_images[check_frame])
        #     ax1.imshow(head_r_images[check_frame])
        #     ax2.imshow(hand_r_images[check_frame])
        #     plt.show()
        #     check_frame = check_frame + check_frame_step

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=str)
    parser.add_argument("--freq", type=float, default=10)
    args = parser.parse_args()

    main(args)