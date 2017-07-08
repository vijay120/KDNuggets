# Detecting Car Lane Lines using Computer Vision

Detecting lane lines is a fundamental task for autonomous vehicles while driving on the road. It is the building block to other path planning and control actions like breaking and steering. Lets get started implementing them!

![Final output of this project](https://i.imgur.com/Oc2hfIz.gif)
*Final output of this project*

Before we work with videos, lets work with static images since it is much easier to debug with. Here is the image we will be working with.

Before we work with videos, lets work with static images since it is much easier to debug with. Here is the image we will be working with.

![](https://cdn-images-1.medium.com/max/1600/1*tcoCQz1m6Wo3e3SeVdxA0g.jpeg)
*Input image*

I am running python 3 with the following imports in a jupyter notebook:

```python
#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
%matplotlib inline
```

The **TLDR** version of the lane detection pipeline is as follows:
1. Pre-process image using grayscale and gaussian blur
1. Apply canny edge detection to the image
1. Apply masking region to the image
1. Apply Hough transform to the image
1. Extrapolate the lines found in the hough transform to construct the left and right lane lines
1. Add the extrapolated lines to the input image

**Step 1: Pre process image**

We grayscale the input image which is needed for canny edge detection.

```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image = mpimg.imread('test_images/solidYellowCurve2.jpg')
# grayscale the image
grayscaled = grayscale(image)
plt.imshow(grayscaled, cmap='gray')
```

![](https://cdn-images-1.medium.com/max/1600/1*HQy8nXyATCAhojIJjL3LMw.png)

*Gray-scaled image*

We then apply a gaussian smoothing function to the image. Again, this is needed for the canny edge detection to average out anomalous gradients in the image.

```python
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
  
# apply gaussian blur
kernelSize = 5
gaussianBlur = gaussian_blur(grayscaled, kernelSize)
```

**Step 2: Canny Edge Detection**

We need to detect edges for lane detection since the contrast between the lane and the surrounding road surface provides us with useful information on detecting the lane lines.

Canny edge detection is an operator that uses the horizontal and vertical gradients of the pixel values of an image to detect edges. A more deeper understanding of the algorithm can be found [here](http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html).

```python
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

# canny
minThreshold = 100
maxThreshold = 200
edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
```

![](https://cdn-images-1.medium.com/max/1600/1*_UoTSNqieJ0NrbcPw9WHlQ.png)

*Notice the edge detector captures all the lane lines, along with surrounding edges like trees*

**Step 3: Mask out points that are not in the region of interest**

The region of interest for the car’s camera is only the two lanes immediately in it’s field of view and not anything extraneous. We can filter out the extraneous pixels by making a polygon region of interest and removing all other pixels that are not in the polygon.

```python
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
  
#apply mask
lowerLeftPoint = [130, 540]
upperLeftPoint = [410, 350]
upperRightPoint = [570, 350]
lowerRightPoint = [915, 540]

pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, lowerRightPoint]], dtype=np.int32)
masked_image = region_of_interest(edgeDetectedImage, pts)
```

![](https://cdn-images-1.medium.com/max/1600/1*wzqUGBAvwBLS1ODOsLj1Jw.png)

*Removed all pixels not in the region of interest*

**Step 4: Hough Transform**

Now that we have detected edges in the region of interest, we want to identify lines which indicate lane lines. This is where the hough transform comes in handy.

The Hough transformation converts a “x vs. y” line to a point in “gradient vs. intercept” space. Points in the image will correspond to lines in hough space. An intersection of lines in hough space will thus correspond to a line in Cartesian space. Using this technique, we can find lines from the pixel outputs of the canny edge detection output. A detailed explanation of the Hough transformation can be found here.

```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.    
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    
#hough lines
rho = 1
theta = np.pi/180
threshold = 30
min_line_len = 20 
max_line_gap = 20

houged = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)
```

![](https://cdn-images-1.medium.com/max/1600/1*ZzCjMl3XC4SWgKjH19PUgg.png)

*The hough transform identifies the lane lines in the image*

**Step 5: Extrapolate the individual, small lines and construct the left and right lane lines**

The hough transform gives us small lines based on the intersections in hough space. Now, we can take this information and construct a global left lane line and a right lane line.

We do this by separating the small lines into two groups, one with a positive gradient and the other with a negative gradient. Due to the depth of the camera’s view, the lane lines slant towards each other as we go further in depth, so that should have the opposite gradients.

We then take the average gradient and intercept values for each group and construct our global lane lines from there. The lane lines are then extrapolated to the edge detected pixel with the minimum y-axis coordinate and the pixel with the maximum y-axis coordinate.

To implement this algorithm, we simply change the draw_lines function stated before to extrapolate the lines.

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    This function draws `lines` with `color` and `thickness`.    
    """
    imshape = img.shape
    
    # these variables represent the y-axis coordinates to which the line will be extrapolated to
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    
    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    
    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)
            
            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]
    
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)
    
    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
    
    # Make sure we have some points in each lane line category
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global), (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), (lower_right_x, ymax_global), color, thickness)
```

![](https://cdn-images-1.medium.com/max/1600/1*5Hdk5YjyFN_1kNc9AgKafw.png)

*Extrapolated lane lines*

**Step 6: Add the extrapolated lines to the input image**

We then overlay the extrapolated lines to the input image. We do this by adding a weight value to the original image based on the detected lane line coordinate.

```python
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
  
# outline the input image
colored_image = weighted_img(houged, image)
```

![](https://cdn-images-1.medium.com/max/1600/1*mYDIxa6gcM6lSYX-XDRfKQ.png)

*Line lanes overlaid on the input image*

Finally, we can add this pipeline of image transformations to videos, frame by frame. For more details on how to do this, the entire implementation of this pipeline and it’s application to videos can be found here.