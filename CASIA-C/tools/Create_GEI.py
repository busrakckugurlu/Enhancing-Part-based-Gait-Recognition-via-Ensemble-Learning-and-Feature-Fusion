# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:29:52 2024

@author: BUSRA
"""



import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from collections import OrderedDict
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images

src_dir= 'D:/Crop_Silh_244_224_CASIA_C'
dest_dir= 'D:/CASIA_C_GEI'

dir_mapping = OrderedDict([(src_dir, dest_dir)])

img_list=[]

for dir, dest_dir in dir_mapping.items():
    print('Processing data in {}'.format(dir))
#ids    
    for index, class_name in enumerate(os.listdir(dir)): 
        class_dir = os.path.join(dir, class_name)
        dest_class_dir = os.path.join(dest_dir, class_name)
        if not os.path.exists(dest_class_dir):
            os.mkdir(dest_class_dir)
            print(dest_class_dir, 'created')
#categories       
        for category in os.listdir(class_dir):
            ctg_dir = os.path.join(class_dir, category)
            dest_ctg_dir = os.path.join(dest_dir, class_name,category)
            if not os.path.exists(dest_ctg_dir):
                os.mkdir(dest_ctg_dir)
        
            for name in os.listdir(ctg_dir):
                path = os.path.join(ctg_dir,name)
                print(path)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img_list.append(img)
                
            result = np.mean(img_list, axis=0)
            Gei_img = result.astype(np.uint8)
            frame_name= class_name + '_' + category + '.png'
            frame_dest_dir = os.path.join(dest_ctg_dir,frame_name)
            cv2.imwrite(frame_dest_dir, Gei_img)
            img_list=[]
                
                # crop_img_head = img[0:45, 80:150]
                # # imshow(crop_img)
                # bigger_crop_head = cv2.resize(crop_img_head, (240, 240))
                
                # frame_name=name
                # frame_dest_dir = os.path.join(dest_ctg_dir,frame_name)
                # cv2.imwrite(frame_dest_dir, bigger_crop_head)
        

#geis

                    