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


'''
=================模板匹配==================
•TM_SQDIFF：计算平方不同，计算出来值越小，越相关
•TM_CCORR：计算相关性，计算出来值越大，越相关
•TM_CCOEFF：计算相关系数，计算出来值越大，越相关
•TM_SQDIFF_NORMED：计算归一化平方不同，越接近0，越相关
•TM_CCORR_NORMED：计算归一化相关性，越接近1，越相关
•TM_CCOEFF_NORMED：计算归一化相关系数，越接近1，越相关
'''

img = cv2.imread('xxx.png', 0)
template = cv2.imread('xxx.png', 0)
h, w = template.shape[:2]

img.shape

template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img, template, 1)  # 1可以变为cv2.TM_CCOEFF
res.shape

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# min_val
# max_val
# min_loc  # 最小值位置左上点
# max_loc

for meth in methods:
    img2 = img.copy()

    # 匹配方法的真值
    method = eval(meth)
    print(method)
    res = cv2.matchTemplate(img, template, method)
    min_val, min_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 如果平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED，取最小值
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 画矩形
    cv2.rectangle(img2, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.subplot(122), plt.imshow(img2, cmap='gray')
    plt.xticks([]), plt.yticks([])  # 隐藏坐标轴
    plt.suptitle(meth)
    plt.show()


img_rgb = cv2.imread('xxxx.png')
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
template = cv2.imread('xxxxx.png',0)
h,w = template.shape[:2]


'''
=================匹配多个==================
'''
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
#取匹配程度大于百分之八十的坐标
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):#*表示可选参数
    bottom_right = (pt[0] + w,pt[1] + h)
    cv2.rectangle(img_rgb,pt,bottom_right,(0,0,255),2)

cv2.imshow('img_rgb',img_rgb)
cv2.waitKey(0)
