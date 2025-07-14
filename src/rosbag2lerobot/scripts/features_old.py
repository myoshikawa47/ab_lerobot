airec_basic_features = {
    'observation.image.head.left': {
        'dtype': 'video',
        # 'shape': [360, 1280, 3],
        'shape': [360, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    'observation.image.arm.right': {
        'dtype': 'video',
        # 'shape': [480, 640, 3],
        'shape': [360, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    # 'observation.state': {
    #     "dtype": "float64",
    #     "shape": [21],
    # },
    'observation.state.steer': {
        "dtype": "float64",
        "shape": (3,),
    },
    'observation.state.right_arm.pos': {
        "dtype": "float64",
        "shape": (9,),
    },
    'observation.state.right_arm.effort': {
        "dtype": "float64",
        "shape": (9,),
    },
    'observation.state': {
        "dtype": "float64",
        "shape": (9,),
    },
    'action.base': {
        "dtype": "float64",
        "shape": (3,),
    },
    'action.right_arm': {
        "dtype": "float64",
        "shape": (9,),
    },
    'action': {
        "dtype": "float64",
        "shape": (12,),
    },
}







hsr_features = {
    "observation.image.head": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    "observation.image.hand": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    "observation.state": {
        "dtype": "float32",
        "shape": [8],
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ],
    },
    "observation.wrench.wrist": {
        "dtype": "float32",
        "shape": [60],
        "names": [
            "force_x_0", "force_y_0", "force_z_0", "torque_x_0", "torque_y_0", "torque_z_0",
            "force_x_1", "force_y_1", "force_z_1", "torque_x_1", "torque_y_1", "torque_z_1",
            "force_x_2", "force_y_2", "force_z_2", "torque_x_2", "torque_y_2", "torque_z_2",
            "force_x_3", "force_y_3", "force_z_3", "torque_x_3", "torque_y_3", "torque_z_3",
            "force_x_4", "force_y_4", "force_z_4", "torque_x_4", "torque_y_4", "torque_z_4",
            "force_x_5", "force_y_5", "force_z_5", "torque_x_5", "torque_y_5", "torque_z_5",
            "force_x_6", "force_y_6", "force_z_6", "torque_x_6", "torque_y_6", "torque_z_6",
            "force_x_7", "force_y_7", "force_z_7", "torque_x_7", "torque_y_7", "torque_z_7",
            "force_x_8", "force_y_8", "force_z_8", "torque_x_8", "torque_y_8", "torque_z_8",
            "force_x_9", "force_y_9", "force_z_9", "torque_x_9", "torque_y_9", "torque_z_9",
        ],
        "description": "Wrist wrench data (force and torque) with 100Hz history (10 samples per frame, flattened)",
    },
    "action.absolute": {
        "dtype": "float32",
        "shape": [8],
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ],
        "description": "absolute action for all joints without hand_motor_joint(gripper)",
    },
    "action.relative": {
        "dtype": "float32",
        "shape": [11],
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
            "base_x",
            "base_y",
            "base_t",
        ],
        "description": "delta action for all joints and base without hand_motor_joint(gripper)",
    },
    "action.arm": {
        "dtype": "float32",
        "shape": [5],
        "names": [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ],
        "description": "absolute action for arm joints",
    },
    "action.gripper": {
        "dtype": "float32",
        "shape": [1],
        "names": ["hand_motor_joint"],
        "description": "absolute action for gripper",
    },
    "action.head": {
        "dtype": "float32",
        "shape": [2],
        "names": ["head_pan_joint", "head_tilt_joint"],
        "description": "absolute action for head joints",
    },
    "action.base": {
        "dtype": "float32",
        "shape": [3],
        "names": ["base_x", "base_y", "base_t"],
        "description": "delta action for base",
    },
    "episode_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "frame_index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
    "timestamp": {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    },
    "next.done": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "index": {
        "dtype": "int64",
        "shape": [1],
        "names": None,
    },
}