# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 21:02:53 2022
this code available in "Chao H, He Y, Zhang J, Feng J (2019) GaitSet: regarding gait as a set for cross-view gait recognition. In: AAAI Conference on Artificial Intelligence (pp. 8126–8133) "
@author: BUSRA
"""

import os
from scipy import misc as scisc
import cv2
import numpy as np
from warnings import warn
from time import sleep

from multiprocessing import Pool
from multiprocessing import TimeoutError as MP_TimeoutError

import imageio

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


INPUT_PATH = 'D:/GaitDatasetC_silh'
OUTPUT_PATH = 'D:/Crop_Silh_244_224_CASIA_C'


IF_LOG = 'False'
LOG_PATH =  './pretreatment.log'
WORKERS = 1
T_H = 224
T_W = 224

def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')
    
    
def cut_img(img, seq_info, frame_name, pid):
    # A silhouette contains too little white pixels
    # might be not valid for identification.
    if img.sum() <= 10000:
        message = 'seq:%s, frame:%s, no data, %d.' % (
            '-'.join(seq_info), frame_name, img.sum())
        warn(message)
        log_print(pid, WARNING, message)
        return None
    # Get the top and bottom point
    y = img.sum(axis=1)
    y_top = (y != 0).argmax(axis=0)
    y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top:y_btm + 1, :]
    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    _r = img.shape[1] / img.shape[0]
    _t_w = int(T_H * _r)
    img = cv2.resize(img, (_t_w, T_H), interpolation=cv2.INTER_CUBIC)
    # Get the median of x axis and regard it as the x center of the person.
    sum_point = img.sum()
    sum_column = img.sum(axis=0).cumsum()
    x_center = -1
    for i in range(sum_column.size):
        if sum_column[i] > sum_point / 2:
            x_center = i
            break
    if x_center < 0:
        message = 'seq:%s, frame:%s, no center.' % (
            '-'.join(seq_info), frame_name)
        warn(message)
        log_print(pid, WARNING, message)
        return None
    h_T_W = int(T_W / 2)
    left = x_center - h_T_W
    right = x_center + h_T_W
    if left <= 0 or right >= img.shape[1]:
        left += h_T_W
        right += h_T_W
        _ = np.zeros((img.shape[0], h_T_W))
        img = np.concatenate([_, img, _], axis=1)
    img = img[:, left:right]
    return img.astype('uint8')


def cut_pickle(seq_info, pid):
    seq_name = '-'.join(seq_info)
    log_print(pid, START, seq_name)
    seq_path = os.path.join(INPUT_PATH, *seq_info)
    out_dir = os.path.join(OUTPUT_PATH, *seq_info)
    frame_list = os.listdir(seq_path)
    frame_list.sort()
    count_frame = 0
    for _frame_name in frame_list:
        frame_path = os.path.join(seq_path, _frame_name)
        img = cv2.imread(frame_path)[:, :, 0]
        img = cut_img(img, seq_info, _frame_name, pid)
        if img is not None:
            # Save the cut img
            save_path = os.path.join(out_dir, _frame_name)
            #scisc.imsave(save_path, img) // hatalı bu
            imageio.imwrite(save_path, img)
            count_frame += 1
    # Warn if the sequence contains less than 5 frames
    if count_frame < 5:
        message = 'seq:%s, less than 5 valid data.' % (
            '-'.join(seq_info))
        warn(message)
        log_print(pid, WARNING, message)

    log_print(pid, FINISH,
              'Contain %d valid frames. Saved to %s.'
              % (count_frame, out_dir))



################################################## CASIA-C Path
results = list()
pid = 0

id_list = os.listdir(INPUT_PATH)
id_list.sort()
# Walk the input path
for _id in id_list:
    seq_type = os.listdir(os.path.join(INPUT_PATH, _id))
    seq_type.sort()
    for _seq_type in seq_type:
        view = os.listdir(os.path.join(INPUT_PATH, _id, _seq_type))
        view.sort()
        seq_info = [_id, _seq_type]
        out_dir = os.path.join(OUTPUT_PATH, *seq_info)
        os.makedirs(out_dir)
        results.append(cut_pickle(seq_info, pid))  
        sleep(0.02)
        pid += 1
        
        
        # for _view in view:
        #     seq_info = [_id, _seq_type, _view]
        #     out_dir = os.path.join(OUTPUT_PATH, *seq_info)
        #     os.makedirs(out_dir)
        #     results.append(cut_pickle(seq_info, pid))                  
        #     sleep(0.02)
        #     pid += 1


# unfinish = 1
# while unfinish > 0:
#     unfinish = 0
#     for i, res in enumerate(results):
#         try:
#             res.get(timeout=0.1)
#         except Exception as e:
#             if type(e) == MP_TimeoutError:
#                 unfinish += 1
#                 continue
#             else:
#                 print('\n\n\nERROR OCCUR: PID ##%d##, ERRORTYPE: %s\n\n\n',i, type(e))
#                 raise e
