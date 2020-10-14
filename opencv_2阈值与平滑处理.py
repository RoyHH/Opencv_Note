import  tensorflow as tf
import cv2 #opencv读取格式RGB
import matplotlib.pyplot as plt #绘图展示
import numpy as np #数值计算工具包
from   tensorflow.keras import layers, optimizers, datasets, Sequential
from   matplotlib import pyplot as plt
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
# cv_show('image',img)
# cv_show('img_noise',img_noise)
# cv_show('image_gray',img_gray)


'''
=================图像阈值处理==================
et,dst = cv2.threshold(src,thresh,maxval,type)

    src：输入图，只能是单通道
    dst：输出图
    thresh：阈值
    maxval：当像素值超过了阈值（小于阈值根据type定）所赋予的值
    type：二值化操作类型：
                        cv2.THRESH_BINARY;超过阈值取最大值，否则为0
                        cv2.THRESH_BINARY_INV;上述反转
                        cv2.THRESH_BINARY_TRUNC;大于阈值部分设为阈值，否则不便
                        cv2.THRESH_TOZERO;大于阈值部分不改变，否则设为0
                        cv2.THRESH_TOZERO_INV.上述反转
'''
# #不同阈值操作并展示
# ret,thresh1 = cv2.threshold(img_gray,127,255, cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img_gray,127,255, cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img_gray,127,255, cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img_gray,127,255, cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img_gray,127,255, cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
#
# for i in range(6):
#     plt.subplot(2, 3, i + 1), plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


'''
=================图像平滑处理==================
'''
# #均值滤波，简单的平均卷积操作，这里是3*3的过滤核
blur = cv2.blur(img_noise,(3,3))
# cv_show('blur',blur)


# #方框滤波，和均值滤波差不多，可以选择归一化
box_norm = cv2.boxFilter(img_noise,-1,(3,3),normalize=True)
# cv_show('box_norm',box_norm)


# #方框滤波，和均值滤波差不多，不选择归一化容易越界，越界指的是大于255
box = cv2.boxFilter(img_noise,-1,(3,3),normalize=False)
# cv_show('box',box)


# #高斯滤波
gaussian = cv2.GaussianBlur(img_noise,(5,5),1)
# cv_show('gaussian',gaussian)


#中值滤波
median = cv2.medianBlur(img_noise,5)
# cv_show('median',median)


# 集中显示对比效果
img_noise = cv_plt(img_noise)
blur = cv_plt(blur)
box_norm = cv_plt(box_norm)
box = cv_plt(box)
gaussian = cv_plt(gaussian)
median = cv_plt(median)
plt.subplot(231), plt.imshow(img_noise), plt.title('img_noise')
plt.subplot(232), plt.imshow(blur), plt.title('img_noise--blur')
plt.subplot(233), plt.imshow(box_norm), plt.title('img_noise--box_norm')
plt.subplot(234), plt.imshow(box), plt.title('img_noise--box')
plt.subplot(235), plt.imshow(gaussian), plt.title('img_noise--gaussian')
plt.subplot(236), plt.imshow(median), plt.title('img_noise--median')
plt.show()