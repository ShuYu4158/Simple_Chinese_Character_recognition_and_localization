# -*- coding: utf-8 -*-
"""
function_version_2  2018.01.05
based on : find_char_2.py
author   : Shu-Yu Li

[input]
File_path  (string)   : input image file path (this image type needs to be surpported by cv2)
Frame_size (int)      : down sampling ratio
Overlap    (int)      : which is overlapped by this and next windows
Gaussian_blur (string): "true" or !"true"

[output]
img      (list)  : 
location (list)  : 
"""
#import library
import numpy as np
import cv2
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt

def sort(img,location,num):
    information =[]
    for i in range(num):
        information.append( [ img[i] , location[i] ] )
    information = sorted(information, key=lambda s: s[1][1])
    information = sorted(information, key=lambda s: s[1][0])
    
    for i in range(num):
        img[i] = information[i][0]
        location[i] = information[i][1]
    
    return img, location


def Find_Char(File_path,Frame_size,Overlap,Gaussian_blur):
    ## load image  
    im = cv2.imread(File_path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if(Gaussian_blur=="true"):
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    
    thresh = 127
    t,im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)
    y, x, c =im.shape
    
    map_buff = np.zeros((int(y/Frame_size),int(x/Frame_size)))
    
    
    ## creat raw figure every 
    # allocate Frame_size*Frame_size pixels to one value
    for i in range(int(y/Frame_size)):
        for j in range(int(x/Frame_size)):
            count=0
            for I in range(Frame_size):
                for J in range(Frame_size):
                    if(im_bw[i*Frame_size+I,j*Frame_size+J]==0):
                        count = count + 1
            map_buff[i,j]=count
    
    
    ## creat feature map ##
    map_result = np.zeros((int(y/Frame_size)-Overlap,int(x/Frame_size)-Overlap))
    for i in range(int(y/Frame_size)-Overlap):
        for j in range(int(x/Frame_size)-Overlap):
            map_result[i,j]=np.sum(map_buff[i:i+Overlap+1,j:j+Overlap+1])
    
    
    ## find local maxima in feature map
    #set parameter
    neighborhood_size = 2
    threshold = np.mean(map_result)
    
    
    ## parameter needed in neighborhood_size converge process ##
    buff_num = 100
    local_maxima_num =0
    
    while (local_maxima_num != buff_num):
        buff_num=local_maxima_num
        data = map_result
        data_max = filters.maximum_filter(data, neighborhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighborhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        
        labeled, num_objects = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
        local_maxima_num, dimension = xy.shape
        neighborhood_size = neighborhood_size+1
    
    
    ## find the corresponding corrodinate based on local maxima in feature map
    local_maxima_num, dimension = xy.shape
    Local_Maxima_Index = xy
    Local_Maxima_Index = Local_Maxima_Index*Frame_size
    Out_Char = []
    Out_Char_lefttop = []
    for i in range(local_maxima_num):
        y_low  = int(Local_Maxima_Index[i,0]-int((neighborhood_size)/2)*Frame_size)
        y_high = int(Local_Maxima_Index[i,0]+(int((neighborhood_size)/2)+4)*Frame_size)
        x_low  = int(Local_Maxima_Index[i,1]-int((neighborhood_size)/2)*Frame_size)
        x_high = int(Local_Maxima_Index[i,1]+(int((neighborhood_size)/2)+4)*Frame_size)
        if (y_low<0):
            y_low = 0
        if (x_low<0):
            x_low = 0
        if (y_high>y):
            y_high = y
        if (x_high>x):
            x_high = x
        Out_Char.append(im_bw[y_low:y_high,x_low:x_high])
        Out_Char_lefttop.append([y_low,x_low])
        #plt.imshow(Out_Char[i])
        #plt.show()
    
    img, location = sort(Out_Char,Out_Char_lefttop,local_maxima_num)
    
    for i in range(local_maxima_num):
        print("sort")
        plt.title(i)
        plt.imshow(img[i])
        plt.show()
        print(location[i])
    
    return img , location
