
import os
import numpy as np
from collections import OrderedDict
import cv2

src_dir= 'D:/Casia_gait/gei'
dest_dir= 'D:/CASIA_WITHOUT_VARIATION'

def remove_object_by_difference(image1, image2, threshold_value=100, condition='nm'):
    
    if condition == 'bg':
        section_ranges = [(0, 40), (40, 90), (90, 140), (140, 190), (190, 240)]
        section_thresholds=[1100,500,380,500,500]
        difference = cv2.absdiff(image1, image2)
        
        _, thresh = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Çantayı çıkarmak için maskeyi orijinal çantalı görüntüye uygulayacağız
        modified_image = np.copy(image1)
        
        # Her parça için fark bölgesini inceleyin
        for i, (start, end) in enumerate(section_ranges):
            # Belirtilen başlangıç ve bitiş satırlarına göre fark bölümünü alın
            section_diff = thresh[start:end, :]
            white_pixel_count = np.sum(section_diff == 255)
    
            if white_pixel_count > section_thresholds[i]:
    
                modified_image[start:end, :] = 0
    
    if condition == 'cl':
        section_ranges = [(0, 40), (40, 90), (90, 140), (140, 190), (190, 240)]
        section_thresholds = [1500, -5, -5, 500, 900]
        
        difference = cv2.absdiff(image1, image2)
        
        # Fark görüntüsünü threshold ile belirginleştir
        _, thresh = cv2.threshold(difference, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Çantayı çıkarmak için maskeyi orijinal çantalı görüntüye uygulayacağız
        modified_image = np.copy(image1)
        
        # Her parça için fark bölgesini inceleyin
        for i, (start, end) in enumerate(section_ranges):
            # Belirtilen başlangıç ve bitiş satırlarına göre fark bölümünü alın
            section_diff = thresh[start:end, :]
            white_pixel_count = np.sum(section_diff == 255)
    
            if white_pixel_count > section_thresholds[i]:
    
                modified_image[start:end, :] = 0
    
    return modified_image, thresh



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
            if category in ["nm-01", "nm-02", "nm-03", "nm-04","nm-05","nm-06"]:
                ctg_dir = os.path.join(class_dir, category)
                dest_ctg_dir = os.path.join(dest_dir, class_name,category)
                if not os.path.exists(dest_ctg_dir):
                    os.mkdir(dest_ctg_dir)
                for name in os.listdir(ctg_dir):
                    path=os.path.join(ctg_dir,name)
                    img = cv2.imread(path)
                    frame_name=name
                    frame_dest_dir = os.path.join(dest_ctg_dir,frame_name)
                    cv2.imwrite(frame_dest_dir, img)
                                    
                    
            elif category in ["bg-01", "bg-02"]:
                ctg_dir = os.path.join(class_dir, category)
                dest_ctg_dir = os.path.join(dest_dir, class_name,category)
                if not os.path.exists(dest_ctg_dir):
                    os.mkdir(dest_ctg_dir)
                    print(dest_ctg_dir, 'created')
                    
                for name in os.listdir(ctg_dir):
                    path=os.path.join(ctg_dir,name)
                    img_bg = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    print(path)
                    
                    split_path = path.split('\\')
                    combined_path = split_path[0] + '/'+ class_name + '/nm-01'
                    img_name = split_path[3].split('-')
                    img_name[1]='nm'
                    img_name[2]='01'
                    image_name = '-'.join(img_name)
                    full_path = combined_path+ '/'+ image_name
                    
                    if not os.path.exists(full_path):
                        combined_path = split_path[0] + '/'+ class_name + '/nm-06'
                        img_name = split_path[3].split('-')
                        img_name[1]='nm'
                        img_name[2]='06'
                        image_name = '-'.join(img_name)
                        full_path = combined_path+ '/'+ image_name
                    
                    img_nm = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
                    
                    result_image, diff_image = remove_object_by_difference(img_bg, img_nm, condition='bg')
                    
                    frame_name=name
                    frame_dest_dir = os.path.join(dest_ctg_dir,frame_name)
                    cv2.imwrite(frame_dest_dir, result_image)
                    

            elif category in ["cl-01", "cl-02"]:
                ctg_dir = os.path.join(class_dir, category)
                dest_ctg_dir = os.path.join(dest_dir, class_name,category)
                if not os.path.exists(dest_ctg_dir):
                    os.mkdir(dest_ctg_dir)
                    print(dest_ctg_dir, 'created')
                    
                for name in os.listdir(ctg_dir):
                    path=os.path.join(ctg_dir,name)
                    img_cl = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                    print(path)
                    
                    split_path = path.split('\\')
                    combined_path = split_path[0] + '/'+ class_name + '/nm-01'
                    img_name = split_path[3].split('-')
                    img_name[1]='nm'
                    img_name[2]='01'
                    image_name = '-'.join(img_name)
                    full_path = combined_path+ '/'+ image_name
                    
                    if not os.path.exists(full_path):
                        combined_path = split_path[0] + '/'+ class_name + '/nm-06'
                        img_name = split_path[3].split('-')
                        img_name[1]='nm'
                        img_name[2]='06'
                        image_name = '-'.join(img_name)
                        full_path = combined_path+ '/'+ image_name
                    
                    img_nm = cv2.imread(full_path,cv2.IMREAD_GRAYSCALE)
                    
                    result_image, diff_image = remove_object_by_difference(img_cl, img_nm,condition='cl')
                    
                    frame_name=name
                    frame_dest_dir = os.path.join(dest_ctg_dir,frame_name)
                    cv2.imwrite(frame_dest_dir, result_image)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
