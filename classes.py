'''
classes
'''

import os
import cv2
import numpy as np
'''
class Calib:
    def __init__(self):
        self.matrix() = np.array([])
        
    def get_matrix(self, chk_brd_img):
        pass
'''
# camera calibration
class Calib:
    def __init__(self):
        self.chess_dir = './calibration/'
        self.chessboard_size = (9,6)
        self.mtx = None
        self.dist = None
        
    def calibration(self):
        '''
        Some pinhole cameras introduce distortion to images. 
        To correct this distortion (camera calibration), we 
        need camera matrix and distortion coefficient. They 
        can be calculated from 20 chessboard images. 
        (6*9 chessboard)
        ref: https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
        '''
        
        objpts = []
        imgpts = []
        
        # prepare object points
        objpt = np.zeros((self.chessboard_size[1]*self.chessboard_size[0], 3), np.float32)
        objpt[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        
        file_list = os.listdir(self.chess_dir)
        file_list = [file for file in file_list if file.endswith('.jpg')]        
        
        for fname in file_list:
            img = cv2.imread(self.chess_dir + fname, cv2.IMREAD_GRAYSCALE)
            
            ret, corners = cv2.findChessboardCorners(img, (self.chessboard_size[0], self.chessboard_size[1]), None)
            
            if ret == True:
                objpts.append(objpt)
                imgpts.append(corners)
            else:
                continue
            
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, img.shape[::-1], None, None)
        
    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

# image preprocessing
class Mask:
    def __init__(self):
        self.img_resize = (1280, 720)
        
        self.gray_img_thresh = 100
        self.Cr_img_thresh = 120
        self.s_img_thresh = 180
        
        self.ksize = 3
        self.sobel_x_thresh = (50, 100)
        self.sobel_y_thresh = (170, 200)
        self.mag_thresh = (60, 250)
        self.ori_thresh = (0.77, 1.2)
    '''
    def get_roi(img, vertices):
        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype = np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        ROI = cv2.bitwise_and(img, img, mask = mask)
        
        return ROI, mask
    '''
    def grad_filtering(self, x_grad, y_grad, mag_thresh, ori_thresh):
            # magnitude of gradients
            gradient_magnitude = np.sqrt(x_grad**2 + y_grad**2)
            normalized_gradient_magnitude = cv2.normalize(gradient_magnitude, None, cv2.NORM_MINMAX).astype(np.uint8)
            
            # orientation of gradients
            gradient_orientation = np.arctan2(np.abs(y_grad), np.abs(x_grad))
            
            # thresholding
            thresholded_magnitude = cv2.inRange(normalized_gradient_magnitude, mag_thresh[0], mag_thresh[1])
            thresholded_orientation = cv2.inRange(gradient_orientation, ori_thresh[0], ori_thresh[1])
            
            result = cv2.bitwise_and(thresholded_magnitude, thresholded_orientation)
            
            return result
        
    def img_preprocessing(self, img):
        img = cv2.resize(img, self.img_resize)
        height, width = img.shape[:2]
        
        gray_img, Cr_img, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
        _, _, s_img = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HLS))
        gray_ = gray_img.copy()
        
        # contrast enhencement: thresholding + normalization
        gray_img[gray_img <= self.gray_img_thresh] = self.gray_img_thresh
        Cr_img[Cr_img <= self.Cr_img_thresh] = self.Cr_img_thresh
        s_img[s_img <= self.s_img_thresh] = self.s_img_thresh
        
        gray_img_normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX)
        Cr_img_normalized = cv2.normalize(Cr_img, None, 0, 255, cv2.NORM_MINMAX)
        s_img_normalized = cv2.normalize(s_img, None, 0, 255, cv2.NORM_MINMAX)
        Cr_img_normalized[gray_ < 50] = 0
        s_img_normalized[gray_ < 50] = 0 # Shadow shows hig saturation
        
        # mask from gradient information
        x_grad_gray = cv2.Sobel(gray_img_normalized, cv2.CV_32F, 1, 0, ksize = self.ksize)
        y_grad_gray = cv2.Sobel(gray_img_normalized, cv2.CV_32F, 0, 1, ksize = self.ksize)
        
        x_grad_Cr = cv2.Sobel(Cr_img_normalized, cv2.CV_32F, 1, 0, ksize = self.ksize)
        y_grad_Cr = cv2.Sobel(Cr_img_normalized, cv2.CV_32F, 0, 1, ksize = self.ksize)
        
        x_grad_s = cv2.Sobel(s_img_normalized, cv2.CV_32F, 1, 0, ksize = self.ksize)
        y_grad_s = cv2.Sobel(s_img_normalized, cv2.CV_32F, 0, 1, ksize = self.ksize)
        
        x_grad_gray_abs = np.abs(x_grad_gray)
        y_grad_gray_abs = np.abs(y_grad_gray)
        
        x_grad_Cr_abs = np.abs(x_grad_Cr)
        y_grad_Cr_abs = np.abs(y_grad_Cr)
        
        x_grad_s_abs = np.abs(x_grad_s)
        y_grad_s_abs = np.abs(y_grad_s)
        
        x_grad_gray_normalized = cv2.normalize(x_grad_gray_abs, None, 0, 255, cv2.NORM_MINMAX)
        y_grad_gray_normalized = cv2.normalize(y_grad_gray_abs, None, 0, 255, cv2.NORM_MINMAX)
        
        x_grad_Cr_normalized = cv2.normalize(x_grad_Cr_abs, None, 0, 255, cv2.NORM_MINMAX)
        y_grad_Cr_normalized = cv2.normalize(y_grad_Cr_abs, None, 0, 255, cv2.NORM_MINMAX)
        
        x_grad_s_normalized = cv2.normalize(x_grad_s_abs, None, 0, 255, cv2.NORM_MINMAX)
        y_grad_s_normalized = cv2.normalize(y_grad_s_abs, None, 0, 255, cv2.NORM_MINMAX)
        
        x_grad_gray_mask = cv2.inRange(x_grad_gray_normalized, self.sobel_x_thresh[0], self.sobel_x_thresh[1])
        y_grad_gray_mask = cv2.inRange(y_grad_gray_normalized, self.sobel_y_thresh[0], self.sobel_y_thresh[1])
        
        x_grad_Cr_mask = cv2.inRange(x_grad_Cr_normalized, self.sobel_x_thresh[0], self.sobel_x_thresh[1])
        y_grad_Cr_mask = cv2.inRange(y_grad_Cr_normalized, self.sobel_y_thresh[0], self.sobel_y_thresh[1])
        
        x_grad_s_mask = cv2.inRange(x_grad_s_normalized, self.sobel_x_thresh[0], self.sobel_x_thresh[1])
        y_grad_s_mask = cv2.inRange(y_grad_s_normalized, self.sobel_y_thresh[0], self.sobel_y_thresh[1])
        
        gray_filtered_mask = self.grad_filtering(x_grad = x_grad_gray, y_grad = y_grad_gray, mag_thresh = self.mag_thresh, ori_thresh = self.ori_thresh)
        Cr_filtered_mask = self.grad_filtering(x_grad = x_grad_Cr, y_grad = y_grad_Cr, mag_thresh = self.mag_thresh, ori_thresh = self.ori_thresh)
        s_filtered_mask = self.grad_filtering(x_grad = x_grad_s, y_grad = y_grad_s, mag_thresh = self.mag_thresh, ori_thresh = self.ori_thresh)
        
        gray_mask = np.zeros_like(gray_img, dtype = np.uint8)
        Cr_mask = np.zeros_like(Cr_img, dtype = np.uint8)
        s_mask = np.zeros_like(s_img, dtype = np.uint8)
        
        gray_mask[(x_grad_gray_mask != 0) | (y_grad_gray_mask != 0) | (gray_filtered_mask != 0)] = 255
        Cr_mask[(x_grad_Cr_mask != 0) | (y_grad_Cr_mask != 0) | (Cr_filtered_mask != 0)] = 255
        s_mask[(x_grad_s_mask != 0) | (y_grad_s_mask != 0) | (s_filtered_mask != 0)] = 255
        
        grad_mask = cv2.bitwise_or(gray_mask, s_mask)
        grad_mask = cv2.bitwise_and(grad_mask, gray_mask) + cv2.bitwise_and(grad_mask, s_mask)
        
        # mask from color information
        color_mask = np.zeros_like(grad_mask)
        color_mask[(gray_img_normalized >= np.percentile(gray_img_normalized, 99)) | (Cr_img_normalized >= np.percentile(Cr_img_normalized, 99))|(s_img_normalized >= 200)] = 255
        
        grad_mask[height - max(height//9,1):, :] = 0
        color_mask[height - max(height//9,1):, :] = 0
        
        return grad_mask, color_mask
    
    def combine(self, mask1, mask2):
    
        combined_mask = cv2.addWeighted(mask1, 1, mask2, 0.7, 0)
        combined_mask_color = np.dstack([np.zeros_like(mask1), mask1, mask2]).astype(np.uint8)
        
        return combined_mask, combined_mask_color

# Lane detection 
class Lane:
    def __init__(self):
        ### only for debugging
        self.debug = False
        ###
        
        self.init_hist_top = np.array([])
        self.init_hist_mid = np.array([])
        self.init_hist_bot = np.array([])
        
        self.detected_left = False
        self.detected_right = False
        
        self.warp_src_vertices = []
        self.warp_dst_vertices = []
        
        self.warped_size = (400,1200)
        self.lane_width = 150 # in pixel
        
        self.M = np.array([])
        self.Minv = np.array([])
        
        self.n_windows = 8        
        self.window_size = ()
        self.detected_pts_limit = 1000
        
        self.left_line_pts       = []
        self.left_line_fit_coeff = []
        
        self.right_line_pts       = []
        self.right_line_fit_coeff = []
        
        self.left_windows_centers = np.array([])
        self.right_windows_centers = np.array([])
        self.new_left_windows_centers = np.array([])
        self.new_right_windows_centers = np.array([])
        
        # ego-position-estimation
        self.left_x_st = []
        self.right_x_st = []

    def set_src(self, img_width, left_bottom, left_top):
        x_min, y_min = left_bottom[0], left_bottom[1]
        x_max, y_max = left_top[0], left_top[1]
        
        lb = (x_min, y_min)
        lt = (x_max, y_max)
        rt = (img_width - x_max, y_max)
        rb = (img_width - x_min, y_min)
        
        src = np.array([lb, lt, rt, rb], dtype = np.float32)
        self.warp_src_vertices = src
    
    def set_dst(self, warp_size):
        self.warp_size = warp_size
        dst = np.float32([(100, warp_size[1]-20), (100, 100), (warp_size[0]-100, 100), (warp_size[0]-100, warp_size[1]-20)])
        self.warp_dst_vertices = dst
    
    def set_perspective_matrix(self):
        M = cv2.getPerspectiveTransform(self.warp_src_vertices, self.warp_dst_vertices)
        Minv = cv2.getPerspectiveTransform(self.warp_dst_vertices, self.warp_src_vertices)
        self.M = M
        self.Minv = Minv
    
    def get_hist_info(self, warped_grad_mask, warped_color_mask):
        height, width = warped_grad_mask.shape[:2]
        step = 3
        grad_ratio = 0.4
        color_ratio = 1 - grad_ratio
        
        
        grad_hist_top = np.sum(warped_grad_mask[height//3:height//3 + (2*height//(3*step)),:], axis = 0)//255
        grad_hist_mid = np.sum(warped_grad_mask[height//3 + (2*height//(3*step)):height//3 + 2*(2*height//(3*step)),:], axis = 0)//255
        grad_hist_bot = np.sum(warped_grad_mask[height//3 + 2*(2*height//(3*step)):,:], axis = 0)//255
        
        color_hist_top = np.sum(warped_color_mask[height//3:height//3 + (2*height//(3*step)),:], axis = 0)//255
        color_hist_mid = np.sum(warped_color_mask[height//3 + (2*height//(3*step)):height//3 + 2*(2*height//(3*step)),:], axis = 0)//255
        color_hist_bot = np.sum(warped_color_mask[height//3 + 2*(2*height//(3*step)):,:], axis = 0)//255
        
        hist_top = grad_ratio * grad_hist_top + color_ratio * color_hist_top
        hist_mid = grad_ratio * grad_hist_mid + color_ratio * color_hist_mid
        hist_bot = grad_ratio * grad_hist_bot + color_ratio * color_hist_bot
        
        if len(self.init_hist_top) == 0:
            self.init_hist_top = hist_top
            self.init_hist_mid = hist_mid
            self.init_hist_bot = hist_bot
        else:
            self.init_hist_top += hist_top
            self.init_hist_mid += hist_mid
            self.init_hist_bot += hist_bot
        
    # initializing center coordinates of windows
    def init_lane_center_point(self, warped_grad_mask, warped_color_mask):
        height, width = warped_grad_mask.shape[:2]
        
        step = 3
        tmp_center_pts_left_x = []
        tmp_center_pts_right_x = []
        tmp_center_pts_y = []
        left_windows_centers = []
        right_windows_centers = []
        
#            pts = []
        tmp_center_pts_left_x.append(np.argmax(self.init_hist_top[:width // 2]))
        tmp_center_pts_left_x.append(np.argmax(self.init_hist_mid[:width // 2]))
        tmp_center_pts_left_x.append(np.argmax(self.init_hist_bot[:width // 2]))
        
        tmp_center_pts_right_x.append(np.argmax(self.init_hist_top[width//2:tmp_center_pts_left_x[0] + self.lane_width + 50]) + width //2)
        tmp_center_pts_right_x.append(np.argmax(self.init_hist_mid[width//2:tmp_center_pts_left_x[1] + self.lane_width + 50]) + width //2)
        tmp_center_pts_right_x.append(np.argmax(self.init_hist_bot[width//2:tmp_center_pts_left_x[2] + self.lane_width + 50]) + width //2)
        
        for i in range(step):            
            tmp_center_pts_y.append(height//3 + i*(2*height//(3*step)) + height//(3*step))
            
#            pts = [tmp_center_pts_left_x, tmp_center_pts_right_x, tmp_center_pts_y]
        left_coeff = np.polyfit(tmp_center_pts_y, tmp_center_pts_left_x, 2)
        right_coeff = np.polyfit(tmp_center_pts_y, tmp_center_pts_right_x, 2)
        
        for j in np.arange(self.n_windows):
            win_height = (2*height//3) // self.n_windows
            y = height//3 + win_height//2 + j*win_height
            left_windows_centers.append((left_coeff[0]* y**2 + left_coeff[1] * y + left_coeff[2] - (win_height//2) * left_coeff[0], y))
            right_windows_centers.append((right_coeff[0]* y**2 + right_coeff[1] * y + right_coeff[2] - (win_height//2) * left_coeff[0], y))
        
        self.left_windows_centers = np.int32(left_windows_centers)
        self.right_windows_centers = np.int32(right_windows_centers)
        
        return np.int32(left_windows_centers), np.int32(right_windows_centers), #np.int32(pts)
    
    def find_lane(self, warped_grad_mask, warped_color_mask):
        x_pts_left = []
        y_pts_left = []
        x_pts_right = []
        y_pts_right = []
        
        self.chk_left = []
        self.chk_right = []
        self.chk_tmp = []
        
        height, width = warped_grad_mask.shape[:2]
        
        self.left_windows_centers = self.new_left_windows_centers
        self.right_windows_centers = self.new_right_windows_centers
        
        window_size = [40, (2*self.warped_size[1]//3)//self.n_windows]
        self.window_size = window_size
        
        if (self.detected_left == False) & (self.detected_right == False):
            left_windows_centers, right_windows_centers = self.init_lane_center_point(warped_grad_mask = warped_grad_mask, warped_color_mask = warped_color_mask)
            
        elif (self.detected_left == True) & (self.detected_right == False):
            _, right_windows_centers = self.init_lane_center_point(warped_grad_mask = warped_grad_mask, warped_color_mask = warped_color_mask)
            left_windows_centers     = self.left_windows_centers
            
        elif (self.detected_left == False) & (self.detected_right == True):
            left_windows_centers, _ = self.init_lane_center_point(warped_grad_mask = warped_grad_mask, warped_color_mask = warped_color_mask)
            right_windows_centers   = self.right_windows_centers
            
        else:
            left_windows_centers  = self.left_windows_centers
            right_windows_centers = self.right_windows_centers
            
        for i in range(self.n_windows):
            tmp_x_left_cand = []
            tmp_y_left_cand = []
            tmp_x_right_cand = []
            tmp_y_right_cand = []
            
            left_window_left_end = left_windows_centers[i][0] - self.window_size[0]//2
            left_window_right_end = left_windows_centers[i][0] + self.window_size[0]//2
            
            right_window_left_end = right_windows_centers[i][0] - self.window_size[0]//2
            right_window_right_end = right_windows_centers[i][0] + self.window_size[0]//2
            
            window_top = left_windows_centers[i][1] - self.window_size[1]//2
            window_bot = left_windows_centers[i][1] + self.window_size[1]//2
            
            if left_window_left_end <= 0:
                left_window_left_end = 0
                left_window_right_end = self.window_size[0]
            elif left_window_right_end >= width:
                left_window_left_end = width - self.window_size[0]
                left_window_right_end = width
            else:
                pass
            
            if right_window_left_end <= 0:
                right_window_left_end = 0
                right_window_right_end = self.window_size[0]
            elif right_window_right_end >= width:
                right_window_left_end = width - self.window_size[0]
                right_window_right_end = width
            else:
                pass
            
            left_grad_roi  = warped_grad_mask[window_top:window_bot, left_window_left_end:left_window_right_end]
            right_grad_roi = warped_grad_mask[window_top:window_bot, right_window_left_end:right_window_right_end]
        	
            left_color_roi  = warped_color_mask[window_top:window_bot, left_window_left_end:left_window_right_end]
            right_color_roi = warped_color_mask[window_top:window_bot, right_window_left_end:right_window_right_end]
            
            left_grad_cand  = np.where(left_grad_roi  != 0)
            right_grad_cand = np.where(right_grad_roi != 0)
        	
            left_color_cand  = np.where(left_color_roi  != 0)
            right_color_cand = np.where(right_color_roi != 0)
        	
            x_left_grad_cand = left_grad_cand[1]
            y_left_grad_cand = left_grad_cand[0]
        
            x_right_grad_cand = right_grad_cand[1]
            y_right_grad_cand = right_grad_cand[0]
            
            x_left_color_cand = left_color_cand[1]
            y_left_color_cand = left_color_cand[0]
        
            x_right_color_cand = right_color_cand[1]
            y_right_color_cand = right_color_cand[0]
            
            # Gradient mask
            if len(x_left_grad_cand) <= 1: # if the line cadidate points on the gradient mask are not detected 
                x_left_grad_cand_filtered = []
                y_left_grad_cand_filtered = []
            else:
                x_left_grad_cand_filtered   = x_left_grad_cand[ (x_left_grad_cand  >= np.percentile(x_left_grad_cand, 5))  & (x_left_grad_cand  <= np.percentile(x_left_grad_cand, 95))]  + left_windows_centers[i][0] - self.window_size[0]//2
                y_left_grad_cand_filtered   = y_left_grad_cand[ (x_left_grad_cand  >= np.percentile(x_left_grad_cand, 5))  & (x_left_grad_cand  <= np.percentile(x_left_grad_cand, 95))]  + left_windows_centers[i][1] - self.window_size[1]//2
            
            if len(x_right_grad_cand) <= 1:
                x_right_grad_cand_filtered = []
                y_right_grad_cand_filtered = []
            else:
                x_right_grad_cand_filtered  = x_right_grad_cand[(x_right_grad_cand >= np.percentile(x_right_grad_cand, 5)) & (x_right_grad_cand <= np.percentile(x_right_grad_cand, 95))] + right_windows_centers[i][0] - self.window_size[0]//2
                y_right_grad_cand_filtered  = y_right_grad_cand[(x_right_grad_cand >= np.percentile(x_right_grad_cand, 5)) & (x_right_grad_cand <= np.percentile(x_right_grad_cand, 95))] + right_windows_centers[i][1] - self.window_size[1]//2
            
            # color mask
            if len(x_left_color_cand) <= 1: # if the line cadidate points on the color mask are not detected 
                x_left_color_cand_filtered = []
                y_left_color_cand_filtered = []
            else:
                x_left_color_cand_filtered  = x_left_color_cand[ (x_left_color_cand  >= np.percentile(x_left_color_cand, 5))  & (x_left_color_cand  <= np.percentile(x_left_color_cand, 95))]  + left_windows_centers[i][0] - self.window_size[0]//2
                y_left_color_cand_filtered  = y_left_color_cand[ (x_left_color_cand  >= np.percentile(x_left_color_cand, 5))  & (x_left_color_cand  <= np.percentile(x_left_color_cand, 95))]  + left_windows_centers[i][1] - self.window_size[1]//2
            
            if len(x_right_color_cand) <= 1:
                x_right_color_cand_filtered = []
                y_right_color_cand_filtered = []
            else:
                x_right_color_cand_filtered = x_right_color_cand[(x_right_color_cand >= np.percentile(x_right_color_cand, 5)) & (x_right_color_cand <= np.percentile(x_right_color_cand, 95))] + right_windows_centers[i][0] - self.window_size[0]//2
                y_right_color_cand_filtered = y_right_color_cand[(x_right_color_cand >= np.percentile(x_right_color_cand, 5)) & (x_right_color_cand <= np.percentile(x_right_color_cand, 95))] + right_windows_centers[i][1] - self.window_size[1]//2
                
            # all line candidate points
            tmp_x_left_cand = np.concatenate([x_left_grad_cand_filtered, x_left_color_cand_filtered])
            tmp_y_left_cand = np.concatenate([y_left_grad_cand_filtered, y_left_color_cand_filtered])
        
            tmp_x_right_cand = np.concatenate([x_right_grad_cand_filtered, x_right_color_cand_filtered])
            tmp_y_right_cand = np.concatenate([y_right_grad_cand_filtered, y_right_color_cand_filtered])
        
            x_pts_left.append(tmp_x_left_cand)
            y_pts_left.append(tmp_y_left_cand)
        
            x_pts_right.append(tmp_x_right_cand)
            y_pts_right.append(tmp_y_right_cand)
              
        # curve fitting with line-candidate-points
        x_pts_left  = np.concatenate(x_pts_left).astype(np.int32)
        x_pts_right = np.concatenate(x_pts_right).astype(np.int32)
        
        y_pts_left  = np.concatenate(y_pts_left).astype(np.int32)
        y_pts_right = np.concatenate(y_pts_right).astype(np.int32)
        
        y_ = np.arange(self.warped_size[1]//3, self.warped_size[1])
        
        if (len(x_pts_left) < self.detected_pts_limit):
            self.detected_left = False
            self.left_line_pts = np.array([])
        else:
            left_coeff = np.polyfit(y_pts_left, x_pts_left, 2)
            x_left_ = np.int32(left_coeff[0] * (y_**2) + left_coeff[1] * y_ + left_coeff[2])
            left_pts = np.vstack([x_left_, y_]).T
            
            self.detected_left = True
            self.left_coeff = left_coeff
            self.left_line_pts = left_pts
            
            if self.debug == True:
                self.left_cand = np.vstack([x_pts_left, y_pts_left]).T
            
        if (len(x_pts_right) < self.detected_pts_limit):
            self.detected_right = False    
            self.right_line_pts = np.array([])
            
        else:
            right_coeff = np.polyfit(y_pts_right, x_pts_right, 2)
            x_right_ = np.int32(right_coeff[0] * (y_**2) + right_coeff[1] * y_ + right_coeff[2])
            right_pts = np.vstack([x_right_, y_]).T
            
            self.detected_right = True
            self.right_coeff = right_coeff
            self.right_line_pts = right_pts

            if self.debug == True:
                self.right_cand = np.vstack([x_pts_right, y_pts_right]).T
        
        # upadate windows center points        
        y_center_pts =  self.left_windows_centers[:,1]
        tmp_x_left = np.int32(self.left_coeff[0] * (y_center_pts ** 2) + self.left_coeff[1]*y_center_pts + self.left_coeff[2])
        self.new_left_windows_centers = np.vstack([tmp_x_left, y_center_pts]).T
            
        tmp_x_right = np.int32(self.right_coeff[0] * (y_center_pts ** 2) + self.right_coeff[1]*y_center_pts + self.right_coeff[2])
        self.new_right_windows_centers = np.vstack([tmp_x_right, y_center_pts]).T
            
    def lane_mask(self):
        line_mask = np.zeros([self.warped_size[1], self.warped_size[0]], dtype = np.uint8)
        lane_mask = np.zeros([self.warped_size[1], self.warped_size[0]], dtype = np.uint8)
        
        if self.detected_left == True:
            left_pts = self.left_line_pts
            cv2.polylines(line_mask, [left_pts], False, 255, 2)
            
        if self.detected_right == True:
            right_pts = self.right_line_pts
            cv2.polylines(line_mask, [right_pts], False, 255, 2)
            
        if (self.detected_left == True) and (self.detected_right == True):
            lane_pts = np.vstack([left_pts, np.flipud(right_pts)])
            cv2.fillPoly(lane_mask, [lane_pts], 255)
        
        return line_mask, lane_mask
    
    def windows_mask(self):
        windows_mask = np.zeros([self.warped_size[1], self.warped_size[0]], dtype = np.uint8)
        left_windows_centers = self.left_windows_centers
        right_windows_centers = self.right_windows_centers
        
        for i in range(self.n_windows):
            window_vertices_left  = np.array([(left_windows_centers[i][0] - self.window_size[0]//2, left_windows_centers[i][1] + self.window_size[1]//2),
                                              (left_windows_centers[i][0] - self.window_size[0]//2, left_windows_centers[i][1] - self.window_size[1]//2),
                                              (left_windows_centers[i][0] + self.window_size[0]//2, left_windows_centers[i][1] - self.window_size[1]//2),
                                              (left_windows_centers[i][0] + self.window_size[0]//2, left_windows_centers[i][1] + self.window_size[1]//2)], dtype = np.int32) # lb, lt, rt, rb
        
            window_vertices_right = np.array([(right_windows_centers[i][0] - self.window_size[0]//2, right_windows_centers[i][1] + self.window_size[1]//2),
                                              (right_windows_centers[i][0] - self.window_size[0]//2, right_windows_centers[i][1] - self.window_size[1]//2), 
                                              (right_windows_centers[i][0] + self.window_size[0]//2, right_windows_centers[i][1] - self.window_size[1]//2),
                                              (right_windows_centers[i][0] + self.window_size[0]//2, right_windows_centers[i][1] + self.window_size[1]//2)], dtype = np.int32) # lb, lt, rt, rb
    
            cv2.polylines(windows_mask, [window_vertices_left], True, 255, 3)
            cv2.polylines(windows_mask, [window_vertices_right], True, 255, 3)
            
        return windows_mask
'''
def draw_on_img(img, mask, color = [0,0,255], beta = 1):
    
    coloring = np.zeros_like(img)
    coloring[mask != 0] = color
    result = cv2.addWeighted(img, 1, coloring, beta, 0)
    
    return result
'''
    

    