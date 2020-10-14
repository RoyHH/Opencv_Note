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
# img = cv2.imread('opencv_cat.jpg')
# img_noise = cv2.imread('opencv_cat_noise.jpg')
# img_gray = cv2.imread('opencv_cat.jpg',cv2.IMREAD_GRAYSCALE)
# img_noise_gray = cv2.imread('opencv_cat_noise.jpg',cv2.IMREAD_GRAYSCALE)

img = cv2.imread('111.jpg')
img_noise = cv2.imread('222.jpg')
img_gray = cv2.imread('111.jpg',cv2.IMREAD_GRAYSCALE)
img_noise_gray = cv2.imread('222.jpg',cv2.IMREAD_GRAYSCALE)
'''
=================Sobel算子==================
cv2.Sobel(src,ddepth,dx,dy,ksize)
 
    ddepth:图像深度(-1,输入深度=输出深度)
    水平dx竖直dy
    ksize是Sobel的大小
'''
# 确认边缘显示是否完整
sobel_x_noconver = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
cv_show('sobel_x_noconver',sobel_x_noconver)

# x方向梯度（水平方向的梯度是右减左）
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
# 绝对值转换（把梯度中负的值做绝对值，取正）
sobel_x = cv2.convertScaleAbs(sobel_x)
# cv_show('sobel_x',sobel_x)

# y方向梯度（水平方向的梯度是下减上）
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)
# cv_show('sobel_y',sobel_y)


# 边缘检测（分开计算），再将x方向与y方向按照权重进行相加求和————不建议这么直接计算，因为效果比较差
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_y = cv2.convertScaleAbs(sobel_y)
sobel_xy_add = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
# cv_show('sobel_xy_add',sobel_xy_add)

# 边缘检测（直接计算），即用sobel将x与y方向梯度联合
sobel_xy = cv2.Sobel(img_gray, cv2.CV_64F, 1, 1, ksize=3)
sobel_xy = cv2.convertScaleAbs(sobel_xy)
# cv_show('sobel_xy',sobel_xy)


'''
=================Scharr算子==================
cv2.Scharr(src, ddepth, dx, dy)

    ddepth:图像深度(-1,输入深度=输出深度)
    水平dx竖直dy
'''
scharr_x = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
scharr_x = cv2.convertScaleAbs(scharr_x)
scharr_y = cv2.convertScaleAbs(scharr_y)
scharr_xy = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
# cv_show('scharr_xy',scharr_xy)


'''
=================Laplacian算子==================
cv2.Scharr(src, ddepth, dx, dy)

    ddepth:图像深度(-1,输入深度=输出深度)
    水平dx竖直dy
'''
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)
# cv_show('laplacian',laplacian)



# 不同算子的差异
# res1 = np.hstack((img_gray, laplacian, sobel_xy_add))
# res2 = np.hstack((sobel_x, sobel_y, sobel_xy))
# res3 = np.hstack((scharr_x, scharr_y, scharr_xy))
# res = np.vstack((res1, res2, res3))
# cv_show('res',res)

titles = ['img_gray','laplacian','sobel_xy_add',
         'sobel_x','sobel_y','sobel_xy',
         'scharr_x','scharr_y','scharr_xy']
images = [img_gray,laplacian,sobel_x_noconver,sobel_xy_add,
          sobel_x,sobel_y,sobel_xy,
          scharr_x,scharr_y,scharr_xy]

for i in range(9):
    plt.subplot(3, 3, i + 1), plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
