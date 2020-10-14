import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 视频读入
cap = cv2.VideoCapture('downstairs_video.mp4')

# 角点检测所需参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7)

# lucas kanade参数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2)  # 窗口大小为15*15，金字塔层数为2

# 随机颜色条
color = np.random.randint(0, 255, (100, 3))

# 拿到第一帧图像
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# 返回所有检测特征点，需要输入图像，角点最大数量（效率），品质因子（特征值越大的越好，来筛选）
# 距离相当于这区间有比这个角点强的，就不要这个弱的了
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)  # 寻找角点

# 创建一个mask
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()  # 这个是取的第二帧图像，上面已经取出了第一帧图像
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 需要传入前一帧和当前图像以及前一帧检测到的角点
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # st=1表示
    good_new = p1[st == 1]
    print(p1.shape)  # (n,1,2)
    print(good_new.shape)  # (n, 2)
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        # new=[692.99805  83.00432]
        a, b = new.ravel()  # 或者[a, b] = new
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # python tolist()方法，将数组或者矩阵转换成列表
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)  # 这个相加不会超出边界

    cv2.imshow('frame', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # 更新
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
