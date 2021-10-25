# 3D Vehicle Tracking
In this project, we want to introduce an innovative approach to track vehicles.
### The scope of the project
In order to track the cars, we are given 3D bounding boxes of them, their orientation and dimensions in the first frame, and the calibration matrix of the camera.  After that, we try to track to find the 3D position and orientation of the vehicles in the following frames. Here you can see the pipeline of the project's implementation.

![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png "Project's flow chart")
###### figure 1

### Detecting visible sides of the vehicle
First of all, we find the visible sides of the vehicle using the angle between the camera and the normal vector of each face of the 3D bounding box.

![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png "Project's flow chart")
###### figure 2-bounding_box
![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png "Project's flow chart")
###### figure 3-angle
![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png "Project's flow chart")

Given the 3D bounding box of the car, we can find point H, the center of one of the bounding box sides, in camera-coordinate. By finding the vector m, the vector between the camera and point H, and vector h, the normal vector of one of the bounding box sides, we can find the inner product of these two vectors. If the inner-product is negative, we will find out that the corresponding side is invisible; otherwise, it is visible.

### Finding Key points in a raw image and filtering them
As you can see in figure 4, in order to find key points in the picture, we used the "fast corner detector" implemented in the "open cv" library. 

![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png "Project's flow chart")
###### figure 4 -finding key points using fast feature detector

Afterward, we generated masks for each side of the car and filtered out those key points that are not on the car in the image.
 
![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png "Project's flow chart")
###### figure 5- filtering out key points

### Finding the 3D position of key points and tracking them using optical flow
Next, we used optical flow in order to find the 2D position of key points in the next frame. In addition, by applying some geometric constraints, we found the 3D position of each key point.

![Alt text](https://github.com/arashasg/3D-car-tracking/blob/create_read_me/Images/flow%20chart.png "Project's flow chart")
###### figure 6 optical flow

### Solving PnP problem
By having the 3D position of each key point and its 2D position on the image, we could find the rotation and translation matrix of the camera by solving the PnP problem. Given the rotation and translation vector of the matrix, we could draw the 3D bounding box in the next frame.

