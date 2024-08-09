import sys
import PID
from create_lanes import get_lanes
# from object_detection import obj_det
import cv2
import time
from imutils.video import VideoStream

from junction import junction
import os

"""
0 means left and 1 means right for lanes.
Shape of a lane: [x1, y1, x2, y2, x_bot, lane side, angle]
"""

SPEED = 70
# Initialise PID
P = 0.25
I = 0.2
D = 0.1
pid = PID.PID(P, I, D) 
pid.SetPoint = 0

def get_heading(prev_heading, img_name, Ev3):
    img = cv2.imread(img_name)
#     stop = obj_det(img_name, img, (__name__=="__main__"), Ev3)
#     if stop:
#         return 0, 0
    junction(img, Ev3, show=(__name__=="__main__"))
    lanes, stop, angle = get_lanes(img, img_name, prev_heading, show=(__name__=="__main__"))
    if stop:
        return 0, 0
    if len(lanes) == 0:
        print("\n\n_________________________________\n\nNO LANES!\n_____________________")
        return SPEED, -prev_heading
    """
    else:
        if len(lanes) == 1:
            print('lane direction: ', lanes[0][-2])
            if abs(angle) <= 15:
                angle *= 75
            else:
                angle *= 1.35
    """

    print("pre pid angle: ", angle)
    pid.update(angle)
    final_heading = round(pid.output, 2)
    return SPEED, final_heading

if __name__=="__main__":
    print("with camera?")
    if input()=="y":
        webcam = VideoStream(0).start()
#         os.system("v4l2-ctl -d /dev/video0 -c auto_exposure=3")
#         os.system("v4l2-ctl -d /dev/video0 -c exposure_time_absolute=20")
        while True:
            cv2.waitKey()
            img = webcam.read()
            cv2.imwrite("frame.jpg", img)
            speed, heading = get_heading(0, 'frame.jpg', None)
            print("speed, heading: ", speed, heading)
    else:
        print(sys.argv[1:][0])
        speed, heading = get_heading(0, sys.argv[1:][0], None)
        print("speed, heading: ", speed, heading)
