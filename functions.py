'''
functions for lane detection
'''

import numpy as np
import cv2
import os

import matplotlib.pyplot as plt

def screen0(img, result):
    size = (420, 240)
    
    img = cv2.resize(img,size)
    result = cv2.resize(result,size)
    
    screen = np.ones((260, 870,3), dtype = np.uint8) * 200
    
    screen[10:250, 10: 430, :] = img
    screen[10:250, 440: 860] = result
    
    return screen

def screen1(img, combined_mask_color, inv_trans_mask_combined, result):
    size = (420, 240)
    
    img = cv2.resize(img, size)
    combined_mask_color = cv2.resize(combined_mask_color, size)
    inv_trans_mask_combined = cv2.resize(inv_trans_mask_combined, size)
    result = cv2.resize(result, size)
    
    screen = np.ones((size[1]*2 + 30, size[0]*2 + 30, 3), dtype = np.uint8) * 200
    
    screen[10:10+size[1], 10:10+size[0],:] = img
    screen[10:10+size[1], 2*10+size[0]:2*(10 + size[0]),:] = result
    screen[2*10+size[1]:2*(10+size[1]), 10:10+size[0],:] = combined_mask_color
    screen[2*10+size[1]:2*(10+size[1]), 2*10+size[0]:2*(10 + size[0]),:] = inv_trans_mask_combined
    
    return screen

def screen2(warped_img, warped_combined_mask_color, lane_mask_combined, warped_result):
    size = (200, 600)
    
    warped_img = cv2.resize(warped_img, size)
    warped_combined_mask_color = cv2.resize(warped_combined_mask_color, size)
    lane_mask_combined = cv2.resize(lane_mask_combined, size)
    warped_result = cv2.resize(warped_result, size)
    
    screen = np.ones((size[1] + 20, size[0]*4 + 50, 3), dtype = np.uint8) * 200
    
    screen[10:10+size[1], 10:10+size[0], :] = warped_img
    screen[10:10+size[1], 10+ 1*(10 + size[0]):2*(10+size[0]), :] = warped_combined_mask_color
    screen[10:10+size[1], 10+ 2*(10 + size[0]):3*(10+size[0]), :] = lane_mask_combined
    screen[10:10+size[1], 10+ 3*(10 + size[0]):4*(10+size[0]), :] = warped_result
    
    return screen

def screen3(img, result, warped_img, warped_result):
    size = (420,240)
    warp_size = (200,600)
    
    img = cv2.resize(img, size)
    result = cv2.resize(result, size)
    warped_img = cv2.resize(warped_img, warp_size)
    warped_result = cv2.resize(warped_result, warp_size)
    
    screen = np.ones((size[1] * 2 + 30, size[0] + 2* warp_size[0] + 40, 3), dtype = np.uint8) * 200
    
    screen[10:10+size[1], 10:10+size[0], :] = img
    screen[20+size[1]:2*(10+size[1]), 10:10+size[0], :] = result
    screen[10:screen.shape[0]-10, 20+size[0]:20+size[0] + warp_size[0], :] = warped_img[110:,:] 
    screen[10:screen.shape[0]-10, 30+size[0]+warp_size[0]:screen.shape[1]-10, :] = warped_result[110:,:]
    
    return screen