import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#cv图片展示通用代码
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间毫秒级，2000ms,0表示任意键终止
    cv2.destroyAllWindows()
#cv颜色通道转成plt颜色通道
def cv_plt(img):
    b, g, r = cv2.split(img)
    img_plt = cv2.merge([r, g, b])
    return img_plt

#检查并读取图像
img = cv2.imread('opencv_cat.jpg')
img_noise = cv2.imread('opencv_cat_noise.jpg')
img_gray = cv2.imread('opencv_cat.jpg',cv2.IMREAD_GRAYSCALE)
img_noise_gray = cv2.imread('opencv_cat_noise.jpg',cv2.IMREAD_GRAYSCALE)

print('img.shape',img.shape)


'''
========================Harris 角点检测==========================
cv2.cornerHarris(src, blockSize, ksize, k)
　　• img：数据类型为 float32 的输入图像。
　　• blockSize：角点检测中要考虑的领域大小。
　　• ksize：Sobel 求导中使用的窗口大小
　　• k：Harris 角点检测方程中的自由参数,取值参数为 [0.04,0.06].
'''
'''
cv2.cvtColor(p1,p2) 是颜色空间转换函数
        p1是需要转换的图片
        p2是转换成何种格式
        cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
        cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
'''
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv_show('gray_img',gray_img)
# # print('gray_img.shape',gray_img.shape)
#
# # 读入图像并转化为float类型，用于传递给harris函数
# gray_img = np.float32(gray_img)
# # print('gray_img.shape',gray_img.shape)
#
# # 对图像执行harris角点检测
# Harris_detector = cv2.cornerHarris(gray_img, 9, 21, 0.04)
#
# # 腐蚀harris结果
# dst = cv2.dilate(Harris_detector, None)
# print('dst.shape',dst.shape)
#
# # 设置阈值
# thres = 0.001 * dst.max()
#
# #焦点颜色设置
# img[dst > thres] = [0, 0, 255]
#
# img = cv_plt(img)
# plt.imshow(img,'gray'),plt.title('img_Harris_detector')
# plt.show()


'''
========================SIFT 尺度不变特征变换算法==========================
SIFT算法的实质是：“不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向” ，
SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，如角点、边缘点、暗区的亮点及亮区的暗点等。
'''
img_small = cv2.resize(img,(0,0), fx = 0.5, fy = 0.5)
img_small_gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

#得到特征点
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(img_gray, None)
keypoints2, descriptor2 = sift.detectAndCompute(img_small_gray, None)

#画角点
img = cv2.drawKeypoints(img,keypoints,img,color=(0, 0, 255))
img_small = cv2.drawKeypoints(img_small,keypoints2,img_small,color=(0, 255, 0))

# img = cv_plt(img)
# img_small = cv_plt(img_small)
# plt.subplot(121),plt.imshow(img,'gray'),plt.title('img_sift')
# plt.subplot(122),plt.imshow(img_small,'gray'),plt.title('img_small_sift')
# plt.show()

cv_show('img_sift', img)
cv_show('img_small_sift', img_small)

'''得到角点以后，可以进一步去做'''
# #计算特征
# keypoints, descriptor = sift.compute(img_gray, keypoints)
# keypoints2, descriptor2 = sift.compute(img_small_gray, keypoints2)