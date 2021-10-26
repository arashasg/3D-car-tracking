# 3D Vehicle Tracking
In this project, we want to introduce an innovative approach to track vehicles.
### The scope of the project
In order to track the cars, we are given 3D bounding boxes of them, their orientation and dimensions in the first frame, and the calibration matrix of the camera.  After that, we try to track to find the 3D position and orientation of the vehicles in the following frames. Here you can see the pipeline of the project's implementation.

<img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png"  width="800px" height="250px"/>
<h6>figure 1</h6>

### Demo
Here's a link to a short video of the project has uploaded.[LINK](https://vimeo.com/manage/videos/639048849 "Link to demo")


### Detecting visible sides of the vehicle
First of all, we find the visible sides of the vehicle using the angle between the camera and the normal vector of each face of the 3D bounding box.

<img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/bounding_box.PNG"  width="600px" height="320px"/>
<h6>figure 2</h6>
<img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/angle.PNG"  width="600px" height="380px"/>
<h6>figure 3</h6>

Given the 3D bounding box of the car, we can find point H, the center of one of the bounding box sides, in camera-coordinate. By finding the vector m, the vector between the camera and point H, and vector h, the normal vector of one of the bounding box sides, we can find the inner product of these two vectors. If the inner-product is negative, we will find out that the corresponding side is invisible; otherwise, it is visible.

### Finding Key points in a raw image and filtering them
As you can see in figure 4, in order to find key points in the picture, we used the "fast corner detector" implemented in the "open cv" library. 

Raw image from KITTI dataset             |  Keypoints detected using FAST feature detection
:-------------------------:|:-------------------------:
<img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/raw_image.png"  width="600px" height="187px"/> |  <img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/keypoints%20on%20picture.png"  width="600px" height="187px"/>
<h6>figure 4</h6>

Afterward, we generated masks for each side of the car and filtered out those key points that are not on the car in the image.
 
<img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/mask%20back.png"  width="414px" height="125px"/>  <img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/mask%20ceiling.png"  width="414px" height="125px"/>
<img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/mask%20left.png"  width="414px" height="125px"/>  <img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/mask%20overal.png"  width="414px" height="125px"/>
<p align="center">
<img src="https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/keypoints%20on%20picture%20filtered.png"  width="600px" height="187px"/>
 </p>
<h6>figure 5</h6>

### Finding the 3D position of key points and tracking them using optical flow
Next, we used optical flow in order to find the 2D position of key points in the next frame. In addition, by applying some geometric constraints, we found the 3D position of each key point. In figure 6, each line connects one of keypoints to its corresponding point in the next frame.


![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/optical%20flow.png)
<h6>figure 6</h6>

### Solving PnP problem
By having the 3D position of each key point and its 2D position on the image, we could find the rotation and translation matrix of the camera by solving the PnP problem. Given the rotation and translation vector of the matrix, we could draw the 3D bounding box in the next frame.

