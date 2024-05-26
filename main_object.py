#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


ovelse = 'squat'
video_path = 'Power Clean (side view).mp4'

# Initialize MediaPipe Pose and Drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Open the video file
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_MSEC, 8e3)

frame_number = 0
csv_data = []
speed_vec = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 9)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow("kuk", cimg)
    """
    #circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
             # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
             # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    """
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#speed_vec = np.array(speed_vec)
#fig, axs = plt.subplots(nrows=2, ncols=2)
# for n , (landmark, ax) in enumerate(zip(landmark_list, axs.ravel())):
#     print(n)
#     ax.plot(speed_vec[n::len(landmark_list)])
#     ax.set_title(landmark)
# plt.show()







