import cv2
import os
from glob import glob
from natsort import natsorted

# def create_video_from_images(image_dir, output_path, fps=10, resize=None):
#     # 画像ファイルを取得（jpg, png対応）
#     image_paths = natsorted(
#         glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))
#     )

#     if not image_paths:
#         raise ValueError("指定されたディレクトリに画像が見つかりません")

#     # 最初の画像でサイズを取得
#     first_img = cv2.imread(image_paths[0])
#     if resize:
#         frame_size = resize
#     else:
#         frame_size = (first_img.shape[1], first_img.shape[0])

#     # 動画ライターを作成
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

#     for img_path in image_paths:
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         if resize:
#             img = cv2.resize(img, resize)
#         video_writer.write(img)

#     video_writer.release()
#     print(f"動画を作成しました: {output_path}")

# # 使用例
# if __name__ == "__main__":
#     image_directory = "../output/20250713_201936"         # 画像ディレクトリ
#     output_video = "../fig/output_video.mp4"    # 出力動画ファイル名
#     create_video_from_images(image_directory, output_video, fps=10)


def create_video_from_images(image_dir, output_path, fps=10, resize=None):
    # 画像ファイルを取得（jpg, png対応）
    image_paths = natsorted(
        glob(os.path.join(image_dir, "*.jpg")) + glob(os.path.join(image_dir, "*.png"))
    )

    if not image_paths:
        raise ValueError("指定されたディレクトリに画像が見つかりません")

    # 最初の画像でサイズを取得
    first_img = cv2.imread(image_paths[0])
    if resize:
        frame_size = resize
    else:
        frame_size = (first_img.shape[1], first_img.shape[0])

    # 動画ライターを作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        if resize:
            # 補完を使わずにリサイズ（最近傍補間）
            img = cv2.resize(img, resize, interpolation=cv2.INTER_NEAREST)
        video_writer.write(img)

    video_writer.release()
    print(f"動画を作成しました: {output_path}")

# 使用例
if __name__ == "__main__":
    image_directory = "../output/20250713_201936"         # 画像ディレクトリ
    output_video = "../fig/output_video.mp4"              # 出力動画ファイル名
    resize_to = (1024, 512)                                 # リサイズするサイズ
    create_video_from_images(image_directory, output_video, fps=10, resize=resize_to)