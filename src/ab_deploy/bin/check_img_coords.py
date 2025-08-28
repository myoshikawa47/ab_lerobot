import cv2

# グローバル変数として座標を保存
click_coords = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: (y={y}, x={x})")
        click_coords.append((y, x))

# 画像読み込み（任意の画像に変更可）
image = cv2.imread("../fig/sample_ab01_img.png")  # RGB or グレースケールどちらでも可
print(image.shape)
cv2.imwrite("../fig/sample_clip_img.png", image[:325, 170:510])

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == 27:  # ESCキーで終了
        break

cv2.destroyAllWindows()


"""
h: 0~325        0~190
w: 170~510      210~540
"""