
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from collections import OrderedDict
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images

src_dir= 'D:/CASIA_C/CASIA_C_GEI'

# dest_dir= 'D:/CASIA_C/5Part/1_PART'
# dest_dir= 'D:/CASIA_C/5Part/2_PART'
# dest_dir= 'D:/CASIA_C/5Part/3_PART'
# dest_dir= 'D:/CASIA_C/5Part/4_PART'
dest_dir= 'D:/CASIA_C/5Part/5_PART'

dir_mapping = OrderedDict([(src_dir, dest_dir)])


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
                path=os.path.join(ctg_dir,name)
                img = cv2.imread(path)
                
                
                # crop_img_PART_1 = img[0:40, 75:145]
                # crop_img_PART_2 = img[40:86, 47:177]
                # crop_img_PART_3 = img[86:132, 47:177]
                # crop_img_PART_4 = img[132:178, 47:177]     
                crop_img_PART_5 = img[178:224, 40:190]  

                
                
                # bigger_crop_PART1 = cv2.resize(crop_img_PART_1, (240, 240))
                # bigger_crop_PART2 = cv2.resize(crop_img_PART_2, (240, 240))
                # bigger_crop_PART3= cv2.resize(crop_img_PART_3, (240, 240))
                # bigger_crop_PART4 = cv2.resize(crop_img_PART_4, (240, 240))
                bigger_crop_PART5 = cv2.resize(crop_img_PART_5, (240, 240))
                
                frame_name=name
                frame_dest_dir = os.path.join(dest_ctg_dir,frame_name)
                
                
                # cv2.imwrite(frame_dest_dir, bigger_crop_PART1)
                # cv2.imwrite(frame_dest_dir, bigger_crop_PART2)
                # cv2.imwrite(frame_dest_dir, bigger_crop_PART3)
                # cv2.imwrite(frame_dest_dir, bigger_crop_PART4)
                cv2.imwrite(frame_dest_dir, bigger_crop_PART5)
        

#geis

                    