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

# img = cv2.imread('opencv_cat.jpg')
# img_gray = cv2.imread('opencv_cat.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('111.jpg')
img_gray = cv2.imread('111.jpg',cv2.IMREAD_GRAYSCALE)
'''
=================canny边缘检测==================
cv2.Canny(image, threshold1, threshold2)

        threshold1: minval下界
        threshold2: maxval上界

只保留上界与下界之间的部分。
'''
# v1 = cv2.Canny(img, 30, 80)
# v2 = cv2.Canny(img, 50, 100)
#
# res = np.hstack((v1, v2))
# cv_show('res',res)


'''
=================高斯金字塔——边缘检测==================
'''
# cv_show('img',img)
# print (img.shape)
#
# #上采样1次
# up = cv2.pyrUp(img)
# cv_show('up',up)
# print (up.shape)
#
# #下采样
# down = cv2.pyrDown(img)
# cv_show('down',down)
# print (down.shape)
#
# #上采样2次
# up2 = cv2.pyrUp(up)
# cv_show('up2',up2)
# print (up2.shape)
#
# #先上采样再下采样
# up = cv2.pyrUp(img)
# up_down = cv2.pyrDown(up)
# cv_show('up_down',up_down)
#
# #对比原图、先上采样再下采样的组合图
# cv_show('img,up_down',np.hstack((img,up_down)))


'''
=================拉普拉斯金字塔——边缘检测==================
Li=img-PyrUp(PyrDown(img))
'''


'''
=================轮廓检测==================
cv2.findContours(img,mode,method)

mode:轮廓检索模式
        •RETR_EXTERNAL：只检索最外侧的轮廓
        •RETR_LIST：检索所有轮廓并保存到一条链表
        •RETR_CCOMP：检索所有轮廓并将他们组织为两层：顶层是各部分的外界边界，第他们组织为两层：顶层是各部分的外界边界，第二层是空洞的边界
        •RETR_TREE：检索所有轮廓，并重构嵌套轮廓的整个层次

method：轮廓逼近方法
        •CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他方式输出多边形（顶点的序列）。
        •CHAIN_APPROX_SIMPLE：压缩水平垂直和斜的部分，函数只保留他们的终点部分。
'''
ret,thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
cv_show('thresh',thresh)

binary,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)


'''
=================绘制轮廓==================
传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度，注意copy
-1所有轮廓，（0，0，255）BGR 4线条宽度
cv2.drawContours(image, contours, contourIdx, color)
'''
draw_img = img.copy()
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),4)
cv_show('res',res)


'''
=================轮廓面积与周长==================
'''
cnt = contours[0]
#面积
cv2.contourArea(cnt)
#周长，True表示闭合的
cv2.arcLength(cnt,True)


'''
=================轮廓近似==================
'''
epsilon = 0.01*cv2.arcLength(cnt,True)#0.1粗糙
approx = cv2.approxPolyDP(cnt,epsilon,True)

res = cv2.drawContours(draw_img,[approx],-1,(0,0,255),4)
cv_show('res',res)


'''
=================轮廓外接矩形==================
'''
area = cv2.contourArea(cnt)
x,y,w,h = cv2.boundingRect(cnt)
img_box = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv_show('img_box',img_box)

rect_area = w*h
extent = float(area)/ rect_area
print('轮廓面积与边界矩形',extent)


'''
=================轮廓外接圆==================
'''
(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img_circle = cv2.circle(img,center,radius,(0,255,0),2)
cv_show('img_circle',img_circle)

