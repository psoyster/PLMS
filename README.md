# PLMS
Parking Lot Monitor System

Using a video file which represents a live stream feed of a security camera.  A seperate file (____filename_____) holds the location of the inspection regions which are the parking spot locations.  

These regions of interest (ROI's) will be blurred using Gaussian function and then their pixel intensities will be evaluated.  A large change in pixel intensity shows that a vehicle is in the spot.
