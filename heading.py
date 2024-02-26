import sys
import PID
import create_lanes
from picamera2 import Picamera2
# from object_detection import obj_det
import cv2
import time

"""
0 means left and 1 means right for lanes.
Shape of a lane: [x1, y1, x2, y2, x_bot, lane side, angle]
"""

SPEED = 60
# Initialise PID
P = 0.2
I = 0.4
D = 0
pid = PID.PID(P, I, D) 
pid.SetPoint = 0

def get_heading(prev_heading, img_name):
    stop = False
    # traffic_signs = obj_det(img_name, show=(__name__=="__main__"))
    # print(f"\n------------------\n{traffic_signs}")
    # for sign in traffic_signs:
    #     if sign[0] == 1:
    #         stop = True
    # if stop:
    #     time.sleep(0.5)
    #     return 0, 0
    lanes, stop, angle = create_lanes.get_lanes(cv2.imread(img_name), img_name, prev_heading, show=(__name__=="__main__"))
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
        picam2 = Picamera2()
        camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
        picam2.configure(camera_config)
        picam2.start()
        while True:
            input()
            picam2.capture_file(f'frame.jpg')
            speed, heading = get_heading(0, 'frame.jpg')
            print("speed, heading: ", speed, heading)
    else:
        print(sys.argv[1:][0])
        speed, heading = get_heading(0, img_name=sys.argv[1:][0])
        print("speed, heading: ", speed, heading)
