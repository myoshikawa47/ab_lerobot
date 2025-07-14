airec_basic_features = {
    'observation.image.head.left': {
        'dtype': 'video',
        # 'shape': [360, 1280, 3],
        # 'shape': [360, 640, 3],
        'shape': [36, 64, 3],
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
        # 'shape': [360, 640, 3],
        'shape': [36, 64, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.fps": 10,
            "video.is_depth_map": "false",
            "has_audio": "false",
        },
    },
    'observation.state': {
        "dtype": "float64",
        "shape": (9,),
    },
    'action': {
        "dtype": "float64",
        "shape": (12,),
    },
}