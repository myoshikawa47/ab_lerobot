#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageOverlayNode:
    def __init__(self):
        rospy.init_node('image_overlay_node')
        self.bridge = CvBridge()

        # クロップ範囲（パラメータとして設定可）
        self.crop_y1 = rospy.get_param("~crop_y1", 0)    # 48
        self.crop_y2 = rospy.get_param("~crop_y2", 0)    # 272
        self.crop_x1 = rospy.get_param("~crop_x1", 0)    # 216
        self.crop_x2 = rospy.get_param("~crop_x2", 0)    # 440

        # ターゲットサイズ（表示画像のサイズ）
        self.target_size = 320

        # .npy画像の読み込み
        # dir_path = rospy.get_param("~dir_path", ".")
        # npy_images = np.load(dir_path + "/param/rs64_right_imgs.npy")
        # npy_images = np.load("../fig/sample_input_img.png")
        # self.overlay_img = npy_images[0, 0]  # (64, 64, 3) assumed
        npy_images = cv2.imread("../fig/sample_input_img.png")
        self.overlay_img = npy_images[:,:,::-1]

        rospy.Subscriber("/zed/zed_node/right/image_rect_color", Image, self.right_img_callback)

        rospy.loginfo("Image Overlay Node Started")
        rospy.spin()

    def right_img_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", str(e))
            return

        # クロップ
        cropped = cv_image  # [self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]

        # サイズチェック（安全対策）
        if cropped.size == 0:
            rospy.logwarn("クロップ範囲が無効です")
            return

        # 両画像を target_size × target_size にリサイズ
        # cropped_resized = cv2.resize(cropped, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        # overlay_resized = cv2.resize(self.overlay_img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        cropped_resized = cropped
        overlay_resized = self.overlay_img
        
        # 重ね合わせ
        blended = cv2.addWeighted(cropped_resized, 0.5, overlay_resized, 0.5, 0)

        # 表示
        cv2.imshow("Overlay Image", blended)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        ImageOverlayNode()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
