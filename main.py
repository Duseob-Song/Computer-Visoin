'''
main 
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from time import time

from classes import *
from functions import *

Lane_mask = Mask()
lines = Lane()
calibration = Calib()

calibration.calibration()

video_name = './vid/project_video.mp4'

VIDEO_SAVE_PATH = './result/'
if not os.path.exists(VIDEO_SAVE_PATH):
    os.makedirs(VIDEO_SAVE_PATH)

INIT_COUNT = 0

if  __name__ == '__main__':
    
    comp_time = []
    video = cv2.VideoCapture(video_name)
    
    VIDEO_SIZE_output = (1280, 720)
    VIDEO_SIZE_0 = (870, 260)
    VIDEO_SIZE_1 = (870, 510)
    VIDEO_SIZE_2 = (850, 620)
    VIDEO_SIZE_3 = (860, 510)
    
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer_output = cv2.VideoWriter(VIDEO_SAVE_PATH + 'lane_detection(output).avi', fourcc, 25.0, VIDEO_SIZE_output)
    writer0 = cv2.VideoWriter(VIDEO_SAVE_PATH + 'lane_detection.avi', fourcc, 25.0, VIDEO_SIZE_0)
    writer1 = cv2.VideoWriter(VIDEO_SAVE_PATH + 'lane_detection(Frame).avi', fourcc, 25.0, VIDEO_SIZE_1)
    writer2 = cv2.VideoWriter(VIDEO_SAVE_PATH + 'lane_detection(Bird-view).avi', fourcc, 25.0, VIDEO_SIZE_2)
    writer3 = cv2.VideoWriter(VIDEO_SAVE_PATH + 'lane_detection(Monitoring).avi', fourcc, 25.0, VIDEO_SIZE_3)
    
    
    while True:
        start = time()
        
        ret, img = video.read()
        
        if ret == False:
            break
        img = calibration.undistort(img = img)
        
        result = img.copy()
        
        height, width = img.shape[:2]
        
        grad_mask, color_mask = Lane_mask.img_preprocessing(img)
        
        lines.set_src(img_width = width, left_bottom = (140, height-80), left_top = (width//2 - 45, height//2 + 80))
        lines.set_dst(warp_size = (400,1200))
        lines.set_perspective_matrix()
        
#        left_bot = (140, height - 80)
#        left_top = (width//2 -45, height//2 +80)
#        right_top = (width//2 +45, height//2 +80)
#        right_bot = (width -140, height -80)
#        
#        vertices = np.array([left_bot, left_top, right_top, right_bot], dtype = np.int32)
        
        warped_img = cv2.warpPerspective(img, lines.M, lines.warp_size)
        warped_grad_mask = cv2.warpPerspective(grad_mask, lines.M, lines.warp_size)
        warped_color_mask = cv2.warpPerspective(color_mask, lines.M, lines.warp_size)
        
        end1 = time()
        
        if INIT_COUNT < 10:
            lines.get_hist_info(warped_grad_mask = warped_grad_mask, warped_color_mask = warped_color_mask)
            INIT_COUNT += 1
        else:
            lines.find_lane(warped_grad_mask = warped_grad_mask, warped_color_mask = warped_color_mask)
            
            line_mask, lane_mask = lines.lane_mask()
            windows_mask = lines.windows_mask()
            
            inv_trans_line_mask = cv2.warpPerspective(line_mask, lines.Minv, (width, height))
            inv_trans_lane_mask = cv2.warpPerspective(lane_mask, lines.Minv, (width, height))
            inv_trans_windows_mask = cv2.warpPerspective(windows_mask, lines.Minv, (width, height))
            inv_trans_windows_mask_c = np.dstack([inv_trans_windows_mask, np.zeros_like(inv_trans_windows_mask), np.zeros_like(inv_trans_windows_mask)])
            
            result[inv_trans_line_mask != 0] = [0,255,0]
            result_output = result.copy()
            result[inv_trans_windows_mask !=0] = [255,0,0]
            
            lane_coloring = np.zeros_like(result)
            lane_coloring[inv_trans_lane_mask != 0] = [255,0,255]
            
            result = cv2.addWeighted(result, 1, lane_coloring, 0.7, 0)
            result_output = cv2.addWeighted(result_output, 1, lane_coloring, 0.7, 0)
            
            # road information
            result_output = lines.disp_road_info(img = result_output)
            
            cv2.rectangle(result_output, (0,700), (1280,720), (0,0,0), -1)
            cv2.putText(result_output, 'Video source: https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4', (18, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))    
            
            writer_output.write(result_output)

            end2 = time()
            
            scr0 = screen0(img, result)
            cv2.rectangle(scr0, (10,0), (860,20), (0,0,0), -1)
            cv2.putText(scr0, 'Video source: https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4', (18, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
            cv2.imshow('Result', scr0)
            writer0.write(scr0)
#            
            combined_mask, combined_mask_color = Lane_mask.combine(mask1 = grad_mask, mask2 = color_mask)
            combined_mask_color = cv2.addWeighted(combined_mask_color, 1, inv_trans_windows_mask_c, 0.9, 0)
            lane_mask_combined = np.dstack([windows_mask, line_mask, np.zeros_like(line_mask)])
            warped_lane_coloring = np.dstack([lane_mask, np.zeros_like(lane_mask), lane_mask])
            lane_mask_combined = cv2.addWeighted(lane_mask_combined, 1, warped_lane_coloring, 0.7, 0)
            
            inv_trans_mask_combined = cv2.warpPerspective(lane_mask_combined, lines.Minv, (width, height))
            scr1 = screen1(img, combined_mask_color, inv_trans_mask_combined, result)
            cv2.rectangle(scr1, (10,0), (860,20), (0,0,0), -1)
            cv2.putText(scr1, 'Video source: https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4', (18, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
            warped_combined_mask, warped_combined_mask_color = Lane_mask.combine(mask1 = warped_grad_mask, mask2 = warped_color_mask)
            windows_mask_c = np.dstack([windows_mask, np.zeros_like(windows_mask), np.zeros_like(windows_mask)])
            warped_combined_mask_color = cv2.addWeighted(warped_combined_mask_color, 1, windows_mask_c, 1, 0)
            
            warped_result = cv2.addWeighted(warped_img, 1, lane_mask_combined, 0.9, 0)
            scr2 = screen2(warped_img, warped_combined_mask_color, lane_mask_combined, warped_result)
            cv2.rectangle(scr2, (10,0), (860,20), (0,0,0), -1)
            cv2.putText(scr2, 'Video source: https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4', (18, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
            scr3 = screen3(img, result, warped_img, warped_result)
            cv2.rectangle(scr3, (10,0), (860,20), (0,0,0), -1)
            cv2.putText(scr3, 'Video source: https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4', (18, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
            
            cv2.imshow('output', result_output)            
            cv2.imshow('On Frame', scr1)
            cv2.imshow('Bird-view', scr2)
            cv2.imshow('Monitoring', scr3)
#
            writer1.write(scr1)
            writer2.write(scr2)
            writer3.write(scr3)            
        
            comp_time.append([end2-start, end1-start, end2-end1]) # [Total computing time, Preprocessing, line detection]
            
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    video.release()
    writer_output.release()
    writer0.release()
    writer1.release()
    writer2.release()
    writer3.release()
    cv2.destroyAllWindows()
    
comp_time = np.array(comp_time)
df = pd.DataFrame({'Total Computing Time': comp_time[:,0],
                   'Image preprecessing': comp_time[:,1],
                   'Lane detection': comp_time[:,2]
                   })
df.to_csv('./result/computing_time.csv', index = False)