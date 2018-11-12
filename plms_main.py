'''


Authors: Paul Soyster
Created: 10/23/2018
Modified: 10/23/2018

*------------------------------------------------------------------------------*

- This program takes a video stream and a yaml file of coordinates that
correspond to individual parking spaces.

- Using the cordinates as "Regions of Interest" (ROI's), an evaluation of the
spot is made to determine wether a car is parked in the spot or if it is vacant.

- Red or green rectangles are then drawn on the video with red indicating an
occupied space and green indicating a non-occupied space.

*------------------------------------------------------------------------------*

'''

import numpy as np
import cv2
import yaml
# import matplotlib.pyplot as plt
# print(cv2.__version__)


# Sources of the video file, yaml file containing spot coordinates,
# and output video file if you want to save it.
vid = r'vid_west_science.mp4'
yaml_data = r"ws_spots.yml"
vid_out = r"C:\Users\psoyster\ET 431\PLMS\plms\outputvid.mp4"


# Maybe just have the thresholds and save?
config = {'laplacian_threshold': 12,
          'save_video': False}

# Video capture from file associated with vid
# Video info used for saved video format
cap = cv2.VideoCapture(vid)

cap.set(cv2.CAP_PROP_POS_FRAMES, 3500)  # jump to frame

video_info = {'fps': cap.get(cv2.CAP_PROP_FPS),
              'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
              'num_of_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}





if config['save_video']:
    fourcc = cv2.VideoWriter_fourcc('C', 'R', 'A','M')
    # options: ('P','I','M','1'), ('D','I','V','X'), ('M','J','P','G'), ('X','V','I','D')
    out = cv2.VideoWriter(vid_out, -1, 25.0,  # video_info['fps'],
                          (video_info['width'], video_info['height']))




'''
Taking coordinates from YAML file and creating inspection regions
'''

with open(yaml_data, 'r') as data:  # reading yaml points for parking spots
    parking_data = yaml.load(data)


parking_contours = []
parking_bounding_rects = []
parking_mask = []


for spot in parking_data:  # using the yaml data
    points = np.array(spot['points'])  # making an array of parking space points

    # rect is the (min x val, min y val, width, height)
    rect = cv2.boundingRect(points)

    points_shifted = points.copy()
    # slicing the point_shifted list using [first:last value in the 0th column]
    points_shifted[:, 0] = points[:, 0] - rect[0]
    # slicing the point_shifted list using [first:last value in the 1st column]
    points_shifted[:, 1] = points[:, 1] - rect[1]


    parking_contours.append(points)  # adding each set of coordinates
    parking_bounding_rects.append(rect)  # adding each drawn rectangle



    mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8),
                            [points_shifted], contourIdx=-1,
                            color=255, thickness=-1, lineType=cv2.LINE_8,)

    mask = (mask == 255)  # Changing to True or False
    parking_mask.append(mask)


parking_status = [False] * len(parking_data)
parking_buffer = [None] * len(parking_data)



'''
Video is being retrieved and used for evaluation
'''

while(cap.isOpened()):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (1280,720))

    frame_out = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#------------------------------------------------------------------------------#
# Gaussian Blur in order to minimize noise on the frames.
    # ksize is the size of the kernel to be used for the convolution.
    # sigmaX is the standard deviation used, if 0 sigmaY is the same

    cv2.GaussianBlur(frame_gray, ksize=(3, 3), sigmaX=0, dst=frame_gray)

#------------------------------------------------------------------------------#


    # ind is the iterator while park is the full dictionary that includes the
    # field 'points' with the [x,y] coordinates
    for ind, park in enumerate(parking_data):
        points = np.array(park['points'])  # same points from YAML file
        rect = parking_bounding_rects[ind]  # using the bounding recangles
        # previously calculated when reading in the YAML data


#------------------------------------------------------------------------------#
# Cropping ROI for faster calculation. looking at each spot one region at a time

# Roi is now set to the pixels inside the inspection region denoted in the
# YAML file

        roi = frame_gray[rect[1]:(rect[1] + rect[3]),
                         rect[0]:(rect[0] + rect[2])]

#------------------------------------------------------------------------------#


#------------------------------------------------------------------------------#
# Laplace function is taking the second derivative of each pixel in the
# region of interest (ROI)
        laplacian = cv2.Laplacian(roi, cv2.CV_64F, ksize=3)


        # x = laplacian * parking_mask[ind]
        # y = np.mean(np.abs(x))
        delta = np.mean(np.abs(laplacian * parking_mask[ind]))


        # True when spot is EMPTY
        # False when spot is OCCUPIED
        status = (delta < config['laplacian_threshold'])



    # Using the status of the spot to determine the color of the rectangle to be
    # drawn on the inspection region
        if status == True:
            color = (0, 255, 0)  # green
            print('Spot',ind,'Delta',delta)
        else:
            color = (0, 0, 255)  # red

    # drawing the bounded rectangle 
        cv2.drawContours(frame_out,  # draw on the image to be shown
                         [points],   # using the YAML points for the points
                         contourIdx=-1,  # use all the contour points
                         color=color,  # GREEN if empty RED is occupied
                         thickness=2,  # thickness of line
                         lineType=cv2.LINE_8)
    if config['save_video']:
        # if video_cur_frame % 35 == 0:  # take every 30 frames
        out.write(frame_out)


    cv2.imshow("PLMS Inspection", frame_out)
    


    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
if config['save_video']:
    out.release()
cv2.destroyAllWindows()


