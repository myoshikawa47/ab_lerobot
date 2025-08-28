import numpy as np
import cv2
import os

def save_video_batch(data, output_dir="output_videos", fps=10, is_rgb=True):
    """
    Args:
        data: numpy array of shape (B, T, H, W, C)
        output_dir: directory to save videos
        fps: frame rate
        is_rgb: if True, converts RGB to BGR for OpenCV
    """
    B, T, H, W, C = data.shape
    os.makedirs(output_dir, exist_ok=True)

    for b in range(B):
        filename = os.path.join(output_dir, f"video_{b:03d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, fps, (W, H))

        for t in range(T):
            frame = data[b, t]

            # 0-1のfloat画像ならuint8に変換（省略可）
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)

            if is_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            writer.write(frame)

        writer.release()
        print(f"Saved: {filename}")