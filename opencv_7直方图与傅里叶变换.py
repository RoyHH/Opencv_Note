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
========================直方图==========================
cv2.calcHist(images,channels,mask,histSize,ranges)

            •image:原图图像格式为uint8/float32，传入图像用[],例如[img]
            •channels:灰度图[0]，彩色图像[0][1][2]BGR。
            •mask:掩模图像
            •histSize:BIN数目
            •ranges:[0-256]
'''
# # 灰度图的直方图显示
# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# hist.shape
# plt.hist(img.ravel(),256)
# plt.show()
#
# # 彩色图的直方图显示
# color = ('b','g','r')
# for i,col in enumerate (color):
#     hist_bgr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(hist_bgr,color = col)
#     plt.xlim([0,256])
# plt.hist(img.ravel(), 256)
# plt.show()


'''
========================直方图中的掩膜mask==========================
'''
# #创建掩膜mask，实际上就是规定一个想要的区域，并将其赋值为255，因为opencv中默认255（白色）是想要的区域
# mask = np.zeros(img.shape[:2],np.uint8)
# mask[150:650,150:500] = 255   #mask[y轴,x轴]
# # cv_show('mask',mask)
#
# #与（&）操作，将原图和掩膜做与操作，得到掩膜区域对应在原图中的部分
# masked_img = cv2.bitwise_and(img,img,mask=mask)
# # cv_show('masked_img',masked_img)
#
# hist_full = cv2.calcHist([img],[2],None,[256],[0,256])
# hist_mask = cv2.calcHist([img],[2],mask,[256],[0,256])
#
# img = cv_plt(img)
# masked_img = cv_plt(masked_img)
#
# plt.subplot(221),plt.imshow(img,'gray')
# plt.subplot(222),plt.imshow(mask,'gray')
# plt.subplot(223),plt.imshow(masked_img,'gray')
# plt.subplot(224),plt.plot(hist_full),plt.plot(hist_mask)
# plt.xlim([0,256])   #限定x轴的显示范围为0~256
# plt.show()


'''
========================直方图均衡化==========================
'''
# #原图直方图
# plt.hist(img_gray.ravel(),256)
# plt.show()
#
# #直方图均衡化，需要注意的是，均衡化处理的是灰度图，直接用bgr图片会报错
# equ = cv2.equalizeHist(img_gray)
# plt.hist(equ.ravel(),256)
# plt.show()
#
# #对比看效果
# plt.subplot(221),plt.imshow(img_gray,'gray'),plt.title('img_gray')
# plt.subplot(222),plt.hist(img_gray.ravel(),256),plt.xlim([0,256])
# plt.subplot(223),plt.imshow(equ,'gray'),plt.title('equ')
# plt.subplot(224),plt.hist(equ.ravel(),256),plt.xlim([0,256])
# plt.show()


'''
========================直方图自适应均衡化==========================
'''
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
res_clahe = clahe.apply(img_gray)

#对比原图、均衡化与自适应均衡化
equ = cv2.equalizeHist(img_gray)
plt.subplot(231),plt.imshow(img_gray,'gray'),plt.title('img_gray')
plt.subplot(234),plt.hist(img_gray.ravel(),256),plt.xlim([0,256])
plt.subplot(232),plt.imshow(equ,'gray'),plt.title('equ')
plt.subplot(235),plt.hist(equ.ravel(),256),plt.xlim([0,256])
plt.subplot(233),plt.imshow(res_clahe,'gray'),plt.title('res_clahe')
plt.subplot(236),plt.hist(res_clahe.ravel(),256),plt.xlim([0,256])
plt.show()


'''
========================傅里叶变换==========================
傅里叶变换的物理意义————把图片转换成由中心低频向外逐渐放射高频的频率图
    •高频 变化剧烈 边界区域
    •低频 变化缓慢 图像内部
    
dft：傅里叶变换
idft：傅里叶逆变换

傅里叶变换作用————滤波：
    •低频滤波器 保留低频，使图像模糊
    •高频滤波器 保留高频，使图像细节增强
    •opencv中主要是cv2.dft()和cv2.idft(),输入图像先转换成np.float32格式。
    •通常频率为0在左上方，shift为转换到中间
    •cv2.dft()返回结果为双通道（实部，虚部）通常需要转换格式（0,255）
'''
# #opencv中规定在做傅里叶变换前，要将图片转换成数据格式
# img_gray_float32 = np.float32(img_gray)
#
# #傅里叶变换
# dft = cv2.dft(img_gray_float32,flags = cv2.DFT_COMPLEX_OUTPUT)
# #将低频的部分拉到中间位置
# dft_shift = np.fft.fftshift(dft)
#
# #傅里叶变换后的数据需要展示，所以将其转换到图片形式，即将实部和虚部映射回图像格式数据，得到灰度图
# magnitude_spectum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
#
# plt.subplot(121),plt.imshow(img_gray,cmap = 'gray')
# plt.title('Input Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectum,cmap = 'gray')
# plt.title('Magnitude Spectum'),plt.xticks([]),plt.yticks([])
# plt.show()


'''
========================傅里叶变换————低频滤波==========================
低频滤波器 保留低频，滤掉高频，也就是说图像中心区域会保持不变，而边界处的图像会变得模糊
'''
# img_gray_float32 = np.float32(img_gray)
#
# dft = cv2.dft(img_gray_float32,flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
#
# #锚定中心位置
# rows,cols = img_gray.shape
# crow,ccol = int(rows/2),int(cols/2)
#
# #低频滤波：在中心点框出60*60的矩形，并赋值为1，作为滤频区域
# mask = np.zeros((rows,cols,2),np.uint8)
# # mask[crow-250:crow+250,ccol-250:ccol+250] = 1
# mask[crow-50:crow+50,ccol-50:ccol+50] = 1
#
# #IDFT：傅里叶逆变换
# fshift = dft_shift*mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#
# plt.subplot(121),plt.imshow(img_gray,cmap = 'gray'),plt.title('Input Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(img_back,cmap = 'gray'),plt.title('Result'),plt.xticks([]),plt.yticks([])
# plt.show()


'''
========================傅里叶变换————高频滤波==========================
高频滤波器 保留高频，滤掉低频，也就是说图像中心区域会被滤掉，而边界处的图像会变得清晰，类似描边
'''
# img_gray_float32 = np.float32(img_gray)
#
# dft = cv2.dft(img_gray_float32,flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
#
# #中心位置
# rows,cols = img_gray.shape
# crow,ccol = int(rows/2),int(cols/2)
#
# #高频滤波
# mask = np.ones((rows,cols,2),np.uint8)
# mask[crow-10:crow+10,ccol-10:ccol+10] = 0
#
# #IDFT
# fshift = dft_shift*mask
# f_ishift = np.fft.ifftshift(fshift)
# img_back = cv2.idft(f_ishift)
# img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
#
# plt.subplot(121),plt.imshow(img_gray,cmap = 'gray'),plt.title('Input Image'),plt.xticks([]),plt.yticks([])
# plt.subplot(122),plt.imshow(img_back,cmap = 'gray'),plt.title('Result'),plt.xticks([]),plt.yticks([])
# plt.show()
