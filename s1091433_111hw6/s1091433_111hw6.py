import cv2
import numpy as np

def RLE_encode(flat_data):
    encode_data=[]
    count = 1
    for i in range(1,len(flat_data)):
        if(flat_data[i]==flat_data[i-1]):
            # 計算同樣color連續出現次數
            count+=1
        else:
            # [color][出現次數]加入encode_data中
            encode_data.append(count)
            encode_data.append(flat_data[i-1])
            count=1
    encode_data.append(count)
    encode_data.append(flat_data[-1])
    return encode_data
    
def RLE_decode(file_name, shape):
    # np.load讀檔
    rle_data=np.load(file_name)
    b_en=rle_data['b_encode']
    g_en=rle_data['g_encode']
    r_en=rle_data['r_encode']
    
    b_de=[]
    g_de=[]
    r_de=[]
    
    # 跟壓縮相反步驟還原檔案
    for i in range(0, len(b_en), 2):
        count=b_en[i]
        color=b_en[i+1]
        b_de.extend([color]*count)
    for i in range(0, len(g_en), 2):
        count=g_en[i]
        color=g_en[i+1]
        g_de.extend([color]*count)
    for i in range(0, len(r_en), 2):
        count=r_en[i]
        color=r_en[i+1]
        r_de.extend([color]*count)
        
    # 將bgr三通道轉換成unit8再用reshape轉換回原圖的尺寸
    b_de=np.array(b_de, dtype=np.uint8).reshape((shape[0],shape[1]))
    g_de=np.array(g_de, dtype=np.uint8).reshape((shape[0],shape[1]))
    r_de=np.array(r_de, dtype=np.uint8).reshape((shape[0],shape[1]))
    # merge再一起
    decompress_img=cv2.merge([b_de,g_de,r_de])
    save_compress(decompress_img)

def save_compress(img):
    # 寫入還原圖片
    # cv2.imwrite(filename,img)
    cv2.imshow('decompress_img',img)
    cv2.waitKey()

def read_img(inputname, outputname):
    img=cv2.imread(inputname)
    cv2.imshow('original image', img)
    # 原圖的解析度大小
    shape=img.shape
    # 將原圖的bgr三通道分離
    b,g,r=cv2.split(img)
    # 各自做壓縮
    b_encode=RLE_encode(np.r_[b.flatten()])
    g_encode=RLE_encode(np.r_[g.flatten()])
    r_encode=RLE_encode(np.r_[r.flatten()])
    # 使用numpy自帶函式將bgr壓縮結果存成同一個壓縮檔
    np.savez_compressed(outputname, b_encode=b_encode, g_encode=g_encode, r_encode=r_encode)
    
    # 還原壓縮檔案
    RLE_decode(outputname, shape)
    
img1 = 's1091433_111hw6/img1.bmp'
img2 = 's1091433_111hw6/img2.bmp'
img3 = 's1091433_111hw6/img3.bmp'
compress_img1='compress_image1.npz'
compress_img2='compress_image2.npz'
compress_img3='compress_image3.npz'

read_img(img1, compress_img1)
print('First image complete\n')
read_img(img2, compress_img2)
print('Second image complete\n')
read_img(img3, compress_img3)
print('Third image complete\n')