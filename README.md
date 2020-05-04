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

## Developement Environment
>OS: Wndows 10 (x64)
>Program Language: Python 3.5
>Image processing library: OpenCV 4.2.0

## Structure
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/Flow_Charts/Structure.png)
>The main pipeline is devided into three main parts: camera calibration, image preprocessing and lane detection
>
>***1) Camera Calibration***- Some pinhole cameras introduce significant distortion to images with an erroneous infotmation. To Correct the distorted image, we need camera matrix and distortion coefficients, and they can be calculated using a set of chessboard images and applied to the video frames.
>   
>***2) Image Preprocessing*** - It's the most important part of this algotithm. First, video frames are transformed into YCrCb or HLS color space and separated into each channel. In this project, Y-, Cr- and S-channels are used. Using thresholding and gradient-based method we can identify plausible line-pixels.
>
>***3) Lane Detection*** - First, we apply perspective transform to preprocessed image to calculate the radius of curvature. We use a sliding-window method to identify the line pixels and to filter out the irrelevant pixels(such as cars or trees). And then lane lines are estimated from the detected line-pixels.
>
## Pipeline
>### 1. Camera calibration
> When a camera takes an image of 3D-object and transforms it into an 2D image, this image can be distorted and deleaver erroneus informations.
> To correct the distortion, we must calculate a camera-matrix and distortion-coeffisients with a set of chessboard images. In this project, cv2.calibrateCamera() function and a set of 20 chessboard images are used to compute these matrix and coefficients.
>
> The code for this process is contained in the ***Calib*** class in the classes.py.
>
>(Further information: [OpenCV Tutorial - Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html))
>
>### 2. Image preprocessing
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/Flow_Charts/Preprocessing.png)

>### 3. Find lane
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/Flow_Charts/Detection.png)

## Result
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/gif/output.gif)
>
> ### 1. Bird-View image
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/gif/Bird_View.gif)
>
> ### 2. Camera image
>![image](https://github.com/DuseobSong/Lane-Detection/blob/master/result/gif/Camera_view.gif)
>
