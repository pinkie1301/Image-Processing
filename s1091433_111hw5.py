import cv2
import numpy as np

def skin_detection(img):
  # 將圖片調整成 HSV 色域
  img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # 在設定範圍內的圖片 pixel 過濾出來做成 mask
  # (0, 20, 70), (50, 255, 255)
  HSV_mask = cv2.inRange(img_HSV, (0, 58, 70), (50, 173, 255))
  # 對圖片作開運算處理掉毛邊
  HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

  # 在 YCbCr 色域中再做一遍
  # (0, 133, 77), (255, 173, 127)
  img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
  YCrCb_mask = cv2.inRange(img_YCrCb, (0, 133, 77), (255, 173, 127))
  YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

  # 將兩個不同色域的 mask 結合成一張過濾較準確的 mask
  combine_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
  combine_mask = cv2.medianBlur(combine_mask, 3)
  combine_mask = cv2.morphologyEx(combine_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
  # 標示遮罩
  red_mask = cv2.cvtColor(combine_mask, cv2.COLOR_GRAY2BGR)
  red_mask = cv2.bitwise_not(red_mask)
  red_mask[combine_mask != 0] = [0, 0, 255]
  # 最終 mask 跟原圖做疊合
  result_merge = cv2.bitwise_and(red_mask, img)

  # 將結果調整成適當大小顯示出來(有需要的話)
  percent = 10
  resize_percent = percent / 100
  dim = (int(img.shape[1] * resize_percent), int(img.shape[0] * resize_percent))
  result_show = cv2.resize(combine_mask, dim, cv2.INTER_NEAREST)
  img_show = cv2.resize(img, dim, cv2.INTER_NEAREST)
  merge_show = cv2.resize(result_merge, dim, cv2.INTER_NEAREST)
  # mask
  cv2.imshow("Mask", result_show)
  # 原圖
  cv2.imshow("Image", img_show)
  # 原圖與mask merge過後
  cv2.imshow("Result", merge_show)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")
img3 = cv2.imread("img3.jpg")
skin_detection(img1)
skin_detection(img2)
skin_detection(img3)