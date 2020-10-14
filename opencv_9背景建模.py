import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 视频读入
cap = cv2.VideoCapture('downstairs_video.mp4')

'''
cv2.getStructuringElement(int shape, Size esize, Point anchor = Point(-1, -1)) 返回指定形状和尺寸的结构元素
        int shape：矩形：MORPH_RECT；交叉形：MORPH_CROSS；椭圆形：MORPH_ELLIPSE
        Size esize：核的大小
        Point anchor = Point(-1, -1)：默认锚点在核中心
'''
# 定义核为椭圆形，大小为3*3
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # 形态学开运算去噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 寻找视频中的轮廓
    im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 计算各轮廓的周长
        perimeter = cv2.arcLength(c, True)
        if perimeter > 180 and perimeter < 280:
            # 找到一个直矩形（不会旋转）
            x, y, w, h = cv2.boundingRect(c)
            # 画出这个矩形
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
