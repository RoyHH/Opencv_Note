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


'''
=================腐蚀操作(内收)==================
'''
# #腐蚀——kernel 为5x5方格，迭代次数2
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)
# erosion_noise = cv2.erode(img_noise,kernel,iterations = 1)
# # cv_show('erosion',erosion)
# # cv_show('erosion_noise',erosion_noise)


# #不同迭代次数对比
# kernel_n = np.ones((5,5),np.uint8)
# erosion_1= cv2.erode(img_noise,kernel_n ,iterations = 1)
# erosion_2= cv2.erode(img_noise,kernel_n ,iterations = 2)
# erosion_3= cv2.erode(img_noise,kernel_n ,iterations = 3)
# #hstask是横向拼接, vstask是纵向拼接
# res = np.hstack((erosion_1,erosion_2,erosion_3))
# # cv_show('res',res)


# # 集中显示对比效果
# img = cv_plt(img)
# erosion = cv_plt(erosion)
# res = cv_plt(res)
# plt.subplot(311), plt.imshow(img_noise), plt.title('img')
# plt.subplot(312), plt.imshow(erosion), plt.title('img--erosion')
# plt.subplot(313), plt.imshow(res), plt.title('img--res')
# plt.show()


'''
=================膨胀操作(外扩)==================
'''
# #膨胀操作
# kernel = np.ones((5,5),np.uint8)
# dilate = cv2.dilate(img,kernel,iterations = 1)
# dilate_noise = cv2.dilate(img_noise,kernel,iterations = 1)
# # cv_show('dilate',dilate)
# # cv_show('dilate_noise',dilate_noise)


# #不同迭代次数
# kernel = np.ones((5,5),np.uint8)
# dilate_1= cv2.dilate(img_noise,kernel,iterations = 1)
# dilate_2= cv2.dilate(img_noise,kernel,iterations = 2)
# dilate_3= cv2.dilate(img_noise,kernel,iterations = 3)
# res = np.hstack((dilate_1,dilate_2,dilate_3))
# # cv_show('res',res)


# # 集中显示对比效果
# img = cv_plt(img)
# dilate = cv_plt(dilate)
# res = cv_plt(res)
# plt.subplot(311), plt.imshow(img), plt.title('img')
# plt.subplot(312), plt.imshow(dilate), plt.title('img--dilate')
# plt.subplot(313), plt.imshow(res), plt.title('img--res')
# plt.show()


'''
=================开运算操作(先内收后外扩)==================
'''
# #开运算：先腐蚀，再膨胀
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# opening_noise = cv2.morphologyEx(img_noise,cv2.MORPH_OPEN,kernel)
# cv_show('opening',opening)
# cv_show('opening_noise',opening_noise)


'''
=================闭运算操作(先外扩后内收)==================
'''
# #闭运算：先膨胀，再腐蚀
# kernel = np.ones((3,3),np.uint8)
# closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
# closing_noise = cv2.morphologyEx(img_noise,cv2.MORPH_CLOSE,kernel)
# cv_show('closing',closing)
# cv_show('closing_noise',closing_noise)


'''
=================梯度运算(外扩减去内收)==================
'''
# #梯度=膨胀-腐蚀
# kernel = np.ones((3,3),np.uint8)
# dilate = cv2.dilate(img,kernel,iterations = 2)
# erosion = cv2.erode(img,kernel,iterations = 2)
#
# dilate_noise = cv2.dilate(img_noise,kernel,iterations = 2)
# erosion_noise = cv2.erode(img_noise,kernel,iterations = 2)
#
# res = np.hstack((dilate,erosion))
# cv_show('res',res)
# res_noise = np.hstack((dilate_noise,erosion_noise))
# cv_show('res_noise',res_noise)
#
# #梯度运算：边界信息
# gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
# cv_show('gradient',gradient)
# gradient_noise = cv2.morphologyEx(img_noise,cv2.MORPH_GRADIENT,kernel)
# cv_show('gradient_noise',gradient_noise)


'''
=================礼帽与黑帽==================
'''
kernel = np.ones((21,21),np.uint8)
#礼帽=原始输入-开运算
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
tophat_noise = cv2.morphologyEx(img_noise,cv2.MORPH_TOPHAT,kernel)
res_TH = np.hstack((tophat,tophat_noise))
cv_show('res_TH',res_TH)

#黑帽=闭运算-原始输入
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
blackhat_noise = cv2.morphologyEx(img_noise,cv2.MORPH_BLACKHAT,kernel)
res_BH = np.hstack((blackhat,blackhat_noise))
cv_show('res_BH',res_BH)


