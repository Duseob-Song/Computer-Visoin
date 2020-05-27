Lane Detection Algorithm
==================

## Objective
The goal of this project is to develope an lane detection algorithm for LDW-systems(Lane Departure Warning Systems).

In this Project, we'll use a gradient-based method and a color-based method to extract information to identify road lanes.

## Files
>* [main.py](https://github.com/DuseobSong/Lane-Detection/blob/master/main.py): main code
>* [classes.py](https://github.com/DuseobSong/Lane-Detection/blob/master/classes.py): contains following classes
> > * class ***Calib***: to get and apply camera-matrix and distortion-coefficients
> > * class ***Mask***: filtering image with gradient-based method and corlor-space-based thresholding
> > * class ***Lane***: perspective transformation, find lane lines with sliding-window method
>* [functions.py](https://github.com/DuseobSong/Lane-Detection/blob/master/functions.py): functions to record results
>* Video source: [https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/project_video.mp4)
## Developement Environment
>OS: Wndows 10 (x64), Python 3.5, OpenCV 4.2.0

## Structure
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/Flow_Charts/Structure.png)
>The main pipeline is devided into three main parts: camera calibration, image preprocessing and lane detection
>
>***1) Camera Calibration***- Some pinhole cameras introduce significant distortion to images with an erroneous infotmation. To Correct the distorted image, we need camera matrix and distortion coefficients, and they can be calculated using a set of chessboard images and applied to the video frames.
>   
>***2) Image Preprocessing*** - It's the most important part of this algotithm. First, video frames are transformed into YCrCb or HLS color space and separated into each channel. In this project, Y-, Cr- and S-channels are used. Using thresholding and gradient-based method we can identify plausible line-pixels.
>
>***3) Lane Detection*** - First, we apply perspective transform to preprocessed image to calculate the radius of curvature. We use a sliding-window method to identify the line pixels and to filter out the irrelevant pixels(such as cars or trees). And then lane lines are estimated from the detected line-pixels.

## Pipeline
>### 1. Camera calibration
> When a camera takes an image of 3D-object and transforms it into an 2D image, this image can be distorted and deliver erroneus informations.
> To correct the distortion, we must calculate a camera-matrix and distortion-coefficients with a set of chessboard images. In this project, [cv2.calibrateCamera( )](https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) function and a set of 20 chessboard images are used to compute these matrix and coefficients.
>
> The code for this process is contained in the ***Calib*** class in the [classes.py](https://github.com/DuseobSong/Lane-Detection/blob/master/classes.py).
>
>(Further information: [OpenCV Tutorial - Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html))
>
> ![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/calibration/calibration.png)

>### 2. Image preprocessing
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/Flow_Charts/Preprocessing.png)
>
> You can find the code for image preprocessing in class ***Mask*** in [classes.py](https://github.com/DuseobSong/Lane-Detection/blob/master/classes.py).
>
> #### ***2-1. Color space transform***
> 
> First, RGB-images are transformed into YCrCb- and HLS color spaces with [cv2.cvtColor( )](https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html). In this project, we use Y-, Cr- and S-channel images to find lane lines. 
>> - ***Y-channel***
>>
>> Y-channel images are similar to gray-scale images, so we can identify line pixels in these images using gradient approach. 
>> ![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/cvtColor/Y.png)
>>
>> However, when the lane line is yellow and the road background color is light gray, it is difficult to distinguish the lane line from the background in gray-scale image. To solve this problem, we need Cr- or S-channel image.
>> ![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/cvtColor/Y_prob.png)
>>
>> - ***S-channel*** 
>>
>>In General, since the lane lines has a high saturation value in clear weather, we can identify lane-line-pixels in S-channel images.
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/cvtColor/S.png) 
>>
>>In S-channel image, the yellow lines on light-gray background can be identified.
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/cvtColor/S_yellow_on_light_gray.png)
>>
>> Note that shadows also have a high saturation value, so pixels with low values in the Y-channel image should be filtered out to identify relevant pixels.
>> <pre><code> S_channel_img[Y_channel_image < shadow_threshold] = 0 </code></pre>
>>***shadow_threshold*** depends on weather condition or ilumination.
>>
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/cvtColor/S_shadow_remove.png)
>>
>> - ***Cr-channel*** 
>>
>>In the Cr-channel image, yellow lines are easily detected, but the white lines are not identified. If the purpose is to detect only the white lines, you can use only Y- and S-channel images.
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/cvtColor/Cr.png)
>>
>> Like S-channel image, yellow lines on light-gray backgraound can be identified in Cr-channel image.
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/cvtColor/Cr_yellow_on_light_gray.png)
>>
>
>#### ***2-2. Contrast enhancement***
>> Here we enhance the image contrast in each channel image to more clearly distinguish line pixels from other objects.
>> First, piexl values lower than threshold are replaced with the threshold, and then the pixels are normalized with the maximum and minimum values.
>><pre><code>channel_img[channel_img < threshold] = threshold
>>channel_img_noramlized = cv2.normalize(channel_img, None, 0, 255, cv2.NORM_MINMAX)</code></pre>
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/contrast_enhancement.png)
>>
>
>#### ***2-3. Filtering with gradient information***
>>In gradient approach, Sobel operator is applied to calculate gradients in x- and y- directions, and then their magnitudes and orientations are calculated. Next, image pixels are thresholded with this gradient informations and the location of filtered image-pixels are saved on a gradient-mask image.
>> You can find the code for this process in the ***img_preprocessing( )*** function of class ***Mask*** in [classes.py](https://github.com/DuseobSong/Lane-Detection/blob/master/classes.py).
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/gradient_filtering.png)
>>
>
>#### ***2-4. Thresholding with color information***
>> Line pixels have high values in the contrast enhanced channel images, and we can filter them with ***np.percentile( )*** function.
>><pre><code>Y_channel_color_mask = np.zeros_like(y_channel_img)
>>Cr_channel_color_mask = np.zeros_like(Cr_channel_img)
>>S_channel_color_mask = np.zeros_like(S_channel_img)
>>
>>Y_channel_color_mask[Y_channel_img >= np.percentile(Y_channel_img_normalized, 99)] = 255
>>Cr_channel_color_mask[Cr_channel_img >= np.percentile(Cr_channel_img_normalized, 99)] = 255
>>S_channel_color_mask[Y_channel_img >= S_color_threshold] = 255
>></code></pre>
>>
>> And then, the masks are combined.
>><pre><code>color_mask = cv2.bitwise_or(Y_channel_color_mask, Cr_channel_color_mask)
>> color_mask = cv2.bitwise_or(color_mask, S_channel_color_mask)</code></pre>
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/color_thresholding.png)
>>
>
>#### Result
>> After the gradient mask and color mask have been calculated, they are combined with ***cv2.bitwise_or( )*** function.
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/Preprocessing%20result.png)
>> 
>
>
>### 3. Find lane
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/Flow_Charts/Detection.png)
>
> You can find the code for lane finding in class ***Lane*** in [classes.py](https://github.com/DuseobSong/Lane-Detection/blob/master/classes.py).
> #### 3.1 Perspective transform
>> To get more information, such as curve direction and radius of curvature, the preprocessed mask image is transformed into bird's-eye view image. 
>> Befor applying perspective transform, we have to define source and destination vertices in the original image and the transformed image. The functions ***set_src( )*** and  ***set_det( )*** in class ***Lane*** represents this process. And then, we can calculate matrices for perspective and invers-perspective transformation with ***set_perspective_matrix( )*** in class ***Lane***. 
>><pre><code>lines = Lane()
>>lines.warp_size = (bird_view_width, bird_view_height)
>>lines.set_src(img_width = img_width, left_bottom = left_bottom_coord, left_top = left_top_coord)
>>lines.set_dst(warp_size = lines.warp_size)
>>lines.set_perspective_matrix()
>></code></pre>
>>
>> Now we can generate Bird's eye view image with these matrices and [cv2.warpPerspectiveTransform( )](https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html) function.
>><pre><code> bird_view_img = cv2.warpPerspectiveTransform(img, M, (lines.warp_size[0], lines.warp_size[1])</code></pre>
>>
>>Here's the result of the perspective transformation.
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/img_preprocessing/perspective_transform/transform.png)
>
>
> #### 3.2 Search windows initialization
>> In this project, I applied sliding-window method to identify and filter out line pixels. The center points of search windows are updated with the previously detected lane lines. However, we need an algorithm to define the initial search-window-center-points.
>> Since the yellow line in the distance is not detected, the ROI is set to the lower 2/3 part of the bird's eye view mask image.
To estimate initial position of the lane lines, the ROI is diviedd into three sub-ROIs. Then, in each sub-ROIs, the appearance frequency histogram of nonzero pixels in the x-direction is calculated, and teh most frequent x-coordinates on the left and right half are estimated as the x-coordinates of the lane lines. 
>>
>> However, if the sub-ROI was set on the empty space of the dashed lines, foreign objects can affect the histogram and it may lead to a erroneous lane estimation. To prevent this, Histograms for each sub-ROIs in the first 10 frames are stacked, and the initial positions of lines are estimated with this stacked histograms.
>><pre><code>if FRAME_COUNT < 10:
>>    lines.get_hist_info(warped_grad_mask = bird_view_grad_mask, warped_color_mask = bird_view_color_mask)
>></code></pre>
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/lane%20detection/s_window_init_stack.png)
>>
>> Foreign objects and vehicle are clearly detected in the mask image, which is calculated from one frame. On the other hand, lane lines are clearly marked in the stacked mask image.
>>
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/lane%20detection/s_window_init.png)
>>
>>Now, we have 6 x-coordinates for lane-line estimation. Y-coordinstes are simply defined as the middle y-coordinates of each sub-ROIs. We can calculate the curve fitting coefficients for initial left and right lines with ***np.polyfit( )*** function. 
>> Eight search windows are assigned to each of the left and right lines, and are placed in ROI one after other in y-direction. And then, we can claculate the initial x-coordinate with the curve fitting coefficients and the y-coordinates of search windows.
>>
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/initialization/init_windows.png)
>>
>> The function ***init_lane_center_point( )*** in class ***Lane*** represents this process. This function only works when the current frame is the first frame or when the lines are not detected in previous frame.
>
>
> #### 3.3 Lane estimation using sliding window method
>>
>>Search windows are predefined in the previous frame, and we only need to consider the pixels inside the windows. Ego-lane-lines can be estimated using curve-fitting with these pixels. The center points of search windows are updated with these lane-lines.
>>Now, we can generate lane mask on bied's-eye view image with these estimated lines. This lane mask is invers-perspective transformed and then applied to source frame.
>>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/initialization/sliding_window.png)

## Result
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/gif/output.gif)
>
> ### 1. Bird's-eye View image
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/gif/Bird_view.gif)
>
> ### 2. Camera view
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/gif/Camera_view.gif)
>
