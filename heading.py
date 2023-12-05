import cv2
import sys
import numpy as np
import math
import PID
from create_lanes import get_lanes
from picamera2 import Picamera2

# Initialise PID
P = 0.65
I = 0.4
D = 0
pid = PID.PID(-P, I, D)
pid.SetPoint = 0

def get_heading(prev_heading, img_name = "frame.jpg"):
    lanes = get_lanes(img_name, "results/")
    angle = 0
    if len(lanes) == 0:
        print("\n\n_________________________________\n\nNO LANES!\n_____________________")
        return -prev_heading
    else:
        for lane in lanes:
            angle += lane[1]
        angle/= len(lanes)

    angle = 90 -angle
    print("pre pid angle: ", angle)
    pid.update(angle)
    final_heading = round(pid.output, 2)
    return final_heading

if __name__=="__main__":
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
    picam2.configure(camera_config)
    picam2.start()
    picam2.capture_file(f'frame.jpg')
    print(get_heading(0))

# if __name__=="__main__":
#     print(get_heading(0, img_name=sys.argv[1:][0]))