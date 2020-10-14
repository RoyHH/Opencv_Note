import  tensorflow as tf
import cv2 #opencv读取格式RGB
import matplotlib.pyplot as plt #绘图展示
import numpy as np #数值计算工具包
from   tensorflow.keras import layers, optimizers, datasets, Sequential
from   matplotlib import pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


'''
=================图片读取==================
'''
# img = cv2.imread('opencv_cat.jpg')
#
# #图像显示,也可以创建多个窗口
# cv2.imshow('image',img)
#
# #等待时间毫秒级，2000ms,0表示任意键终止
# cv2.waitKey(2000)
# cv2.destroyAllWindows()
#
# #获取图像长宽 h w c(BGR三色图)
# img.shape
#
# #cv2.IMREAD_COLOR：彩色图像; cv2.IMREAD_GRAYSCALE：灰度图像
# img = cv2.imread('opencv_cat.jpg',cv2.IMREAD_GRAYSCALE)
# #保存变换的灰度图片
# cv2.imwrite('opencv_cat_gray.jpg',img)


'''
=================视频读取==================
'''
# video = cv2.VideoCapture('downstairs_video.mp4')
#
# #检查是否打开正确
# if video.isOpened():
#     # 从第一帧到最后一帧的结果，布尔类型的值
#     open, frame = video.read()
# else:
#     open = False
#
# #遍历每一帧
# while open:
#     ret, frame = video.read()
#     if frame is None:
#         break
#         # 每帧画面转换成灰度图像，并按帧进行显示
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('result', gray)
#         # 每帧画面播放10ms，且q为退出键
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
# video.release()
# cv2.destroyAllWindows()

'''
=================截取图像==================
'''
#截取部分图像数据
# def cv_show(name,img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)  # 等待时间毫秒级，2000ms,0表示任意键终止
#     cv2.destroyAllWindows()
#
# img = cv2.imread('opencv_cat.jpg')
# img = img[100:500, 300:500]
# cv_show('cat', img)


'''
=================颜色通道提取==================
'''
# def cv_show(name,img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)  # 等待时间毫秒级，2000ms,0表示任意键终止
#     cv2.destroyAllWindows()
#
# img = cv2.imread('opencv_cat.jpg')
# b,g,r = cv2.split(img)
#
# #只保留R通道
# cur_img = img.copy()
# cur_img[:,:,0] = 0
# cur_img[:,:,1] = 0
# cv_show('R',cur_img)
#
# #只保留G通道
# cur_img = img.copy()
# cur_img[:,:,0] = 0
# cur_img[:,:,2] = 0
# cv_show('G',cur_img)
#
# #只保留B通道
# cur_img = img.copy()
# cur_img[:,:,1] = 0
# cur_img[:,:,2] = 0
# cv_show('B',cur_img)


'''
=================边界填充==================
'''
#上下左右分别填充的大小
# top_size,bottom_size,left_size,right_size = (50,100,150,200)
#
# img = cv2.imread('opencv_cat.jpg')
#
# replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
# constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
#
# #subplot(231)表示subplot(2，3，1)，即这个figure就是个2*3的矩阵图，也就是总共有6个图，1就代表了第一幅图
# plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
# plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE') #复制边缘像素
# plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT') #反射法 dcba|abcd|dcba
# plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101') #反射法 bcd|abcd|cba
# plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP') #反射法 bcd|abcd|abc
# plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
# plt.show()


'''
=================图像融合==================
'''
# img_cat = cv2.imread('opencv_cat.jpg')
# img_dog = cv2.imread('opencv_dog.jpg')
# img_cat.shape
# img_dog.shape
#
# # 重置图像大小，确保图像大小一致
# img_cat = cv2.resize(img_cat,(500,500))
# img_dog = cv2.resize(img_dog,(500,500))
# img_cat.shape
# img_dog.shape
#
# #图像融合比例3：7，其中0是亮度级提量，偏置量
# res = cv2.addWeighted(img_cat,0.4,img_dog,0.6,0)
# plt.imshow(res)

# plt.show()

'''
=================图像重置==================
'''
# img_cat = cv2.imread('opencv_cat.jpg')
# img_cat1 = cv2.imread('opencv_cat.jpg')
#
# # 重置图像大小到600*500
# res = cv2.resize(img_cat,(600,500))
# # 按比例（1.5：1）重置图像大小
# res1 = cv2.resize(img_cat1,(0,0), fx = 1.5, fy = 1)
#
# plt.subplot(121), plt.imshow(res), plt.title('size')
# plt.subplot(122), plt.imshow(res1), plt.title('scale')
# plt.show()


'''
=================opencv与plt显示中颜色通道顺序不同==================
'''
def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间毫秒级，2000ms,0表示任意键终止
    cv2.destroyAllWindows()

img_cat = cv2.imread('opencv_cat.jpg')
#颜色通道处理。opencv的接口使用BGR，而plt则是RGB模式，不做处理的话，颜色通道匹配不上，会导致显示图片颜色失真（灰度图像）
b, g, r = cv2.split(img_cat)
img_cat1 = cv2.merge([r, g, b])

plt.subplot(121), plt.imshow(img_cat), plt.title('plt_image_bgr')
plt.subplot(122), plt.imshow(img_cat1), plt.title('plt_image_rgb')
plt.show()

cv_show('bgr image', img_cat)
cv_show('rgb image', img_cat1)



